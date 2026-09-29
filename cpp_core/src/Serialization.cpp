#include "rclib/Serialization.h"

#include <cstring>
#include <ios>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

namespace {

constexpr char magic[8] = {'R', 'C', 'L', 'I', 'B', 'M', 'D', 'L'};
constexpr std::int64_t max_count = std::numeric_limits<std::int32_t>::max();

using StorageIndex = Eigen::SparseMatrix<double>::StorageIndex;
static_assert(sizeof(StorageIndex) == sizeof(std::int32_t), "sparse indices are stored as i32");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "doubles are stored as IEEE-754 f64");

void requireLittleEndian() {
  const std::uint16_t probe = 1;
  unsigned char first_byte = 0;
  std::memcpy(&first_byte, &probe, 1);
  if (first_byte != 1) {
    throw SerializationError("rclib model files are only supported on little-endian hosts.");
  }
}

} // namespace

std::int64_t sumFeatureWidths(const std::vector<std::int64_t> &widths) {
  std::int64_t total = 0;
  for (const std::int64_t width : widths) {
    // Each width is an int, so the running total cannot overflow int64 before this check trips.
    total += width;
    if (total > max_count - 1) {
      throw SerializationError("Model: the combined reservoir output width is too large.");
    }
  }
  return total;
}

BinaryWriter::BinaryWriter(std::ostream &os) : os(os) { requireLittleEndian(); }

void BinaryWriter::writeHeader() {
  writeBytes(magic, sizeof(magic));
  writeUInt(serialization_format_version);
}

void BinaryWriter::writeBool(bool value) { writeU8(value ? 1 : 0); }

void BinaryWriter::writeU8(std::uint8_t value) { writeBytes(&value, sizeof(value)); }

void BinaryWriter::writeInt(std::int32_t value) { writeBytes(&value, sizeof(value)); }

void BinaryWriter::writeUInt(std::uint32_t value) { writeBytes(&value, sizeof(value)); }

void BinaryWriter::writeDouble(double value) { writeBytes(&value, sizeof(value)); }

void BinaryWriter::writeInt64(std::int64_t value) { writeBytes(&value, sizeof(value)); }

void BinaryWriter::writeString(const std::string &value) {
  if (value.size() > serialization_max_string_length) {
    throw SerializationError("string is too long to serialize.");
  }
  writeUInt(static_cast<std::uint32_t>(value.size()));
  writeBytes(value.data(), value.size());
}

void BinaryWriter::writeMatrix(const Eigen::MatrixXd &matrix) {
  if (matrix.rows() > max_count || matrix.cols() > max_count) {
    throw SerializationError("matrix is too large to serialize.");
  }
  writeInt64(matrix.rows());
  writeInt64(matrix.cols());
  writeBytes(matrix.data(), static_cast<std::size_t>(matrix.size()) * sizeof(double));
}

void BinaryWriter::writeSparse(const Eigen::SparseMatrix<double> &matrix) {
  if (!matrix.isCompressed()) {
    Eigen::SparseMatrix<double> compressed = matrix;
    compressed.makeCompressed();
    writeSparse(compressed);
    return;
  }
  const Eigen::Index cols = matrix.cols();
  const Eigen::Index nnz = matrix.nonZeros();
  if (matrix.rows() > max_count || cols >= max_count || nnz > max_count) {
    throw SerializationError("sparse matrix is too large to serialize.");
  }
  writeInt64(matrix.rows());
  writeInt64(cols);
  writeInt64(nnz);
  writeBytes(matrix.outerIndexPtr(), static_cast<std::size_t>(cols + 1) * sizeof(StorageIndex));
  writeBytes(matrix.innerIndexPtr(), static_cast<std::size_t>(nnz) * sizeof(StorageIndex));
  writeBytes(matrix.valuePtr(), static_cast<std::size_t>(nnz) * sizeof(double));
}

void BinaryWriter::writeBytes(const void *data, std::size_t size) {
  if (size == 0) {
    return;
  }
  try {
    os.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  } catch (const std::ios_base::failure &e) {
    throw SerializationError(std::string("failed to write model data: ") + e.what());
  }
  if (!os) {
    throw SerializationError("failed to write model data.");
  }
}

BinaryReader::BinaryReader(std::istream &is) : is(is), remaining(0) {
  requireLittleEndian();
  try {
    const std::streampos start = is.tellg();
    if (start == std::streampos(-1)) {
      throw SerializationError("model input stream must be seekable.");
    }
    is.seekg(0, std::ios::end);
    const std::streampos end = is.tellg();
    is.seekg(start);
    if (!is || end == std::streampos(-1) || end < start) {
      throw SerializationError("model input stream must be seekable.");
    }
    remaining = static_cast<std::uint64_t>(end - start);
  } catch (const std::ios_base::failure &e) {
    throw SerializationError(std::string("model input stream must be seekable: ") + e.what());
  }
}

void BinaryReader::readHeader() {
  char file_magic[sizeof(magic)];
  readBytes(file_magic, sizeof(file_magic));
  if (std::memcmp(file_magic, magic, sizeof(magic)) != 0) {
    throw SerializationError("not an rclib model file (bad magic bytes).");
  }
  const std::uint32_t version = readUInt();
  if (version == 0 || version > serialization_format_version) {
    throw SerializationError("unsupported model format version " + std::to_string(version) +
                             " (this build reads versions 1.." + std::to_string(serialization_format_version) + ").");
  }
  format_version = version;
}

bool BinaryReader::readBool() {
  const std::uint8_t value = readU8();
  if (value > 1) {
    throw SerializationError("invalid boolean value " + std::to_string(value) + ".");
  }
  return value == 1;
}

std::uint8_t BinaryReader::readU8() {
  std::uint8_t value = 0;
  readBytes(&value, sizeof(value));
  return value;
}

std::int32_t BinaryReader::readInt() {
  std::int32_t value = 0;
  readBytes(&value, sizeof(value));
  return value;
}

std::uint32_t BinaryReader::readUInt() {
  std::uint32_t value = 0;
  readBytes(&value, sizeof(value));
  return value;
}

double BinaryReader::readDouble() {
  double value = 0.0;
  readBytes(&value, sizeof(value));
  return value;
}

std::int64_t BinaryReader::readInt64() {
  std::int64_t value = 0;
  readBytes(&value, sizeof(value));
  return value;
}

std::int64_t BinaryReader::readCount(const char *what) {
  const std::int64_t value = readInt64();
  if (value < 0 || value > max_count) {
    throw SerializationError(std::string("invalid ") + what + " " + std::to_string(value) + ".");
  }
  return value;
}

std::string BinaryReader::readString() {
  const std::uint32_t length = readUInt();
  if (length > serialization_max_string_length) {
    throw SerializationError("string length " + std::to_string(length) + " exceeds the maximum of " +
                             std::to_string(serialization_max_string_length) + ".");
  }
  std::string value(length, '\0');
  readBytes(&value[0], length);
  return value;
}

Eigen::MatrixXd BinaryReader::readMatrix() {
  const std::int64_t rows = readCount("matrix row count");
  const std::int64_t cols = readCount("matrix column count");
  // rows * cols < 2^62, so the product cannot overflow; compare entries, not bytes.
  if (static_cast<std::uint64_t>(rows * cols) > remaining / sizeof(double)) {
    throw SerializationError("matrix size exceeds the remaining model data.");
  }
  Eigen::MatrixXd matrix(rows, cols);
  readBytes(matrix.data(), static_cast<std::size_t>(matrix.size()) * sizeof(double));
  return matrix;
}

Eigen::SparseMatrix<double> BinaryReader::readSparse() {
  const std::int64_t rows = readCount("sparse row count");
  const std::int64_t cols = readCount("sparse column count");
  const std::int64_t nnz = readCount("sparse nonzero count");
  if (cols == max_count) {
    throw SerializationError("invalid sparse column count " + std::to_string(cols) + ".");
  }
  // The outer index array is payload too, so every allocation below is bounded
  // by the bytes left in the input. At most 2^35 bytes: no overflow.
  const std::uint64_t payload = static_cast<std::uint64_t>(cols + 1) * sizeof(StorageIndex) +
                                static_cast<std::uint64_t>(nnz) * (sizeof(StorageIndex) + sizeof(double));
  if (payload > remaining) {
    throw SerializationError("sparse matrix size exceeds the remaining model data.");
  }

  Eigen::SparseMatrix<double> matrix(rows, cols);
  matrix.resizeNonZeros(nnz);
  StorageIndex *outer = matrix.outerIndexPtr();
  StorageIndex *inner = matrix.innerIndexPtr();
  readBytes(outer, static_cast<std::size_t>(cols + 1) * sizeof(StorageIndex));
  readBytes(inner, static_cast<std::size_t>(nnz) * sizeof(StorageIndex));
  readBytes(matrix.valuePtr(), static_cast<std::size_t>(nnz) * sizeof(double));

  // Validate the whole outer array before using it to index the inner array.
  if (outer[0] != 0 || outer[cols] != nnz) {
    throw SerializationError("invalid sparse matrix structure (outer index bounds).");
  }
  for (std::int64_t j = 0; j < cols; ++j) {
    if (outer[j] > outer[j + 1]) {
      throw SerializationError("invalid sparse matrix structure (decreasing outer index).");
    }
  }
  for (std::int64_t j = 0; j < cols; ++j) {
    for (StorageIndex k = outer[j]; k < outer[j + 1]; ++k) {
      if (inner[k] < 0 || inner[k] >= rows || (k > outer[j] && inner[k] <= inner[k - 1])) {
        throw SerializationError("invalid sparse matrix structure (row indices out of range or not increasing).");
      }
    }
  }
  return matrix;
}

void BinaryReader::readBytes(void *data, std::size_t size) {
  if (size > remaining) {
    throw SerializationError("unexpected end of model data.");
  }
  if (size == 0) {
    return;
  }
  try {
    is.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
  } catch (const std::ios_base::failure &e) {
    throw SerializationError(std::string("failed to read model data: ") + e.what());
  }
  if (!is || static_cast<std::size_t>(is.gcount()) != size) {
    throw SerializationError("unexpected end of model data.");
  }
  remaining -= size;
}

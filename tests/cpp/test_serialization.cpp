#include "rclib/Serialization.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <catch2/catch_all.hpp>
#include <cstdint>
#include <cstring>
#include <functional>
#include <istream>
#include <limits>
#include <new>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;

namespace {

const double nan_value = std::numeric_limits<double>::quiet_NaN();
const double inf_value = std::numeric_limits<double>::infinity();
constexpr std::int64_t int32_max = std::numeric_limits<std::int32_t>::max();

// Appends the raw (little-endian host) bytes of `value`, for crafting malformed input.
template <typename T> void appendRaw(std::string &bytes, T value) {
  char buffer[sizeof(T)];
  std::memcpy(buffer, &value, sizeof(T));
  bytes.append(buffer, sizeof(T));
}

// Writes a sparse matrix in the on-disk layout without any validation.
std::string craftSparse(std::int64_t rows, std::int64_t cols, const std::vector<std::int32_t> &outer,
                        const std::vector<std::int32_t> &inner) {
  std::string bytes;
  appendRaw(bytes, rows);
  appendRaw(bytes, cols);
  appendRaw(bytes, static_cast<std::int64_t>(inner.size()));
  for (std::int32_t value : outer) {
    appendRaw(bytes, value);
  }
  for (std::int32_t value : inner) {
    appendRaw(bytes, value);
  }
  for (std::size_t k = 0; k < inner.size(); ++k) {
    appendRaw(bytes, 1.0);
  }
  return bytes;
}

// memcmp that accepts the null data pointers of empty Eigen objects.
bool sameBytes(const void *a, const void *b, std::size_t size) { return size == 0 || std::memcmp(a, b, size) == 0; }

bool sameBits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

bool sameBits(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         sameBytes(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(double));
}

// Compressed layout (outer/inner index arrays and values) is identical, bit for bit.
bool sameLayout(const Eigen::SparseMatrix<double> &a, const Eigen::SparseMatrix<double> &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols() || a.nonZeros() != b.nonZeros() || !a.isCompressed() ||
      !b.isCompressed()) {
    return false;
  }
  const auto nnz = static_cast<std::size_t>(a.nonZeros());
  return std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.cols() + 1, b.outerIndexPtr()) &&
         std::equal(a.innerIndexPtr(), a.innerIndexPtr() + nnz, b.innerIndexPtr()) &&
         sameBytes(a.valuePtr(), b.valuePtr(), nnz * sizeof(double));
}

// A stream buffer that serves `data` but cannot seek, like a pipe.
class NonSeekableBuffer : public std::streambuf {
public:
  explicit NonSeekableBuffer(std::string data) : data(std::move(data)) {
    setg(&this->data[0], &this->data[0], &this->data[0] + this->data.size());
  }

private:
  std::string data;
};

// A stream buffer whose every write fails, like a full disk.
class FailingBuffer : public std::streambuf {
protected:
  int_type overflow(int_type /*ch*/) override { return traits_type::eof(); }
};

} // namespace

TEST_CASE("Serialization - scalars and strings round-trip", "[serialization]") {
  std::stringstream buffer;
  BinaryWriter writer(buffer);
  writer.writeBool(true);
  writer.writeBool(false);
  writer.writeU8(255);
  writer.writeInt(std::numeric_limits<std::int32_t>::min());
  writer.writeInt(-1);
  writer.writeUInt(std::numeric_limits<std::uint32_t>::max());
  writer.writeDouble(-0.0);
  writer.writeDouble(nan_value);
  writer.writeDouble(-inf_value);
  writer.writeString("");
  writer.writeString(std::string(serialization_max_string_length, 'x'));

  BinaryReader reader(buffer);
  REQUIRE(reader.readBool());
  REQUIRE_FALSE(reader.readBool());
  REQUIRE(reader.readU8() == 255);
  REQUIRE(reader.readInt() == std::numeric_limits<std::int32_t>::min());
  REQUIRE(reader.readInt() == -1);
  REQUIRE(reader.readUInt() == std::numeric_limits<std::uint32_t>::max());
  REQUIRE(sameBits(reader.readDouble(), -0.0));
  REQUIRE(sameBits(reader.readDouble(), nan_value));
  REQUIRE(sameBits(reader.readDouble(), -inf_value));
  REQUIRE(reader.readString().empty());
  REQUIRE(reader.readString() == std::string(serialization_max_string_length, 'x'));
  // Everything was consumed.
  REQUIRE_THROWS_MATCHES(reader.readU8(), SerializationError, MessageMatches(ContainsSubstring("end of model data")));
}

TEST_CASE("Serialization - dense matrices round-trip bit-exactly", "[serialization]") {
  Eigen::MatrixXd values = Eigen::MatrixXd::Random(3, 4);
  values(0, 1) = nan_value;
  values(2, 3) = inf_value;
  values(1, 0) = -0.0;
  const std::vector<Eigen::MatrixXd> matrices = {Eigen::MatrixXd(0, 0), Eigen::MatrixXd(0, 3), Eigen::MatrixXd(2, 0),
                                                 values};

  std::stringstream buffer;
  BinaryWriter writer(buffer);
  for (const auto &matrix : matrices) {
    writer.writeMatrix(matrix);
  }
  BinaryReader reader(buffer);
  for (const auto &matrix : matrices) {
    REQUIRE(sameBits(reader.readMatrix(), matrix));
  }
}

TEST_CASE("Serialization - sparse matrices round-trip with an identical layout", "[serialization]") {
  Eigen::SparseMatrix<double> compressed(6, 5);
  compressed.insert(0, 0) = 1.5;
  compressed.insert(4, 0) = -2.0;
  compressed.insert(2, 3) = 0.0; // explicit zeros are part of the layout and must be kept
  compressed.insert(5, 4) = nan_value;
  compressed.makeCompressed();
  Eigen::SparseMatrix<double> uncompressed(4, 4);
  uncompressed.insert(1, 2) = 3.0; // still in uncompressed mode after insert()

  const std::vector<Eigen::SparseMatrix<double>> matrices = {
      Eigen::SparseMatrix<double>(0, 0), Eigen::SparseMatrix<double>(3, 3), compressed, uncompressed};

  std::stringstream buffer;
  BinaryWriter writer(buffer);
  for (const auto &matrix : matrices) {
    writer.writeSparse(matrix);
  }
  BinaryReader reader(buffer);
  for (const auto &matrix : matrices) {
    Eigen::SparseMatrix<double> expected = matrix;
    expected.makeCompressed();
    REQUIRE(sameLayout(reader.readSparse(), expected));
  }
}

TEST_CASE("Serialization - header", "[serialization]") {
  std::string bytes;

  SECTION("Valid header") {
    std::stringstream buffer;
    BinaryWriter(buffer).writeHeader();
    BinaryReader reader(buffer);
    REQUIRE(reader.formatVersion() == 0);
    reader.readHeader();
    REQUIRE(reader.formatVersion() == serialization_format_version);
  }

  SECTION("Bad magic bytes") {
    bytes = "RCLIBXXX";
    appendRaw(bytes, serialization_format_version);
    std::istringstream input(bytes);
    BinaryReader reader(input);
    REQUIRE_THROWS_MATCHES(reader.readHeader(), SerializationError, MessageMatches(ContainsSubstring("magic")));
  }

  SECTION("Unsupported versions") {
    const std::uint32_t version = GENERATE(0U, serialization_format_version + 1);
    bytes = "RCLIBMDL";
    appendRaw(bytes, version);
    std::istringstream input(bytes);
    BinaryReader reader(input);
    REQUIRE_THROWS_MATCHES(reader.readHeader(), SerializationError,
                           MessageMatches(ContainsSubstring("unsupported model format version")));
  }

  SECTION("Truncated header") {
    std::istringstream input("RCLIB");
    BinaryReader reader(input);
    REQUIRE_THROWS_AS(reader.readHeader(), SerializationError);
  }
}

TEST_CASE("Serialization - malformed primitives are rejected", "[serialization]") {
  using Decoder = std::function<void(BinaryReader &)>;
  const Decoder read_bool = [](BinaryReader &reader) { reader.readBool(); };
  const Decoder read_string = [](BinaryReader &reader) { reader.readString(); };
  const Decoder read_matrix = [](BinaryReader &reader) { reader.readMatrix(); };
  const Decoder read_sparse = [](BinaryReader &reader) { reader.readSparse(); };

  std::string bytes;
  Decoder decode;

  SECTION("Boolean byte other than 0 or 1") {
    appendRaw<std::uint8_t>(bytes, 2);
    decode = read_bool;
  }
  SECTION("String longer than the maximum") {
    appendRaw(bytes, serialization_max_string_length + 1);
    bytes += std::string(serialization_max_string_length + 1, 'x');
    decode = read_string;
  }
  SECTION("String longer than the remaining input") {
    appendRaw<std::uint32_t>(bytes, 10);
    bytes += "abc";
    decode = read_string;
  }
  SECTION("Negative matrix dimension") {
    appendRaw<std::int64_t>(bytes, -1);
    appendRaw<std::int64_t>(bytes, 1);
    decode = read_matrix;
  }
  SECTION("Matrix dimension above INT32_MAX") {
    appendRaw<std::int64_t>(bytes, int32_max + 1);
    appendRaw<std::int64_t>(bytes, 1);
    decode = read_matrix;
  }
  SECTION("Huge matrix with a tiny payload is rejected before allocating") {
    appendRaw<std::int64_t>(bytes, int32_max);
    appendRaw<std::int64_t>(bytes, int32_max);
    appendRaw(bytes, 1.0);
    decode = read_matrix;
  }
  SECTION("Matrix with a truncated payload") {
    appendRaw<std::int64_t>(bytes, 2);
    appendRaw<std::int64_t>(bytes, 2);
    for (int i = 0; i < 3; ++i) {
      appendRaw(bytes, 1.0);
    }
    decode = read_matrix;
  }
  SECTION("Sparse matrix with a huge column count and no nonzeros") {
    // Only the (cols + 1)-element outer index array would be allocated.
    appendRaw<std::int64_t>(bytes, 1);
    appendRaw<std::int64_t>(bytes, int32_max - 1);
    appendRaw<std::int64_t>(bytes, 0);
    decode = read_sparse;
  }
  SECTION("Sparse column count of INT32_MAX") {
    appendRaw<std::int64_t>(bytes, 1);
    appendRaw<std::int64_t>(bytes, int32_max);
    appendRaw<std::int64_t>(bytes, 0);
    decode = read_sparse;
  }
  SECTION("Sparse structure") {
    decode = read_sparse;
    SECTION("Outer index not starting at zero") { bytes = craftSparse(3, 2, {1, 1, 2}, {0, 1}); }
    SECTION("Outer index ending before nnz") { bytes = craftSparse(3, 2, {0, 1, 1}, {0, 1}); }
    SECTION("Outer index decreasing past the inner array") { bytes = craftSparse(3, 2, {0, 100, 2}, {0, 1}); }
    SECTION("Row index out of range") { bytes = craftSparse(3, 2, {0, 1, 2}, {0, 3}); }
    SECTION("Row index negative") { bytes = craftSparse(3, 2, {0, 1, 2}, {-1, 0}); }
    SECTION("Row indices not increasing") { bytes = craftSparse(3, 2, {0, 2, 2}, {2, 1}); }
    SECTION("Duplicate row index") { bytes = craftSparse(3, 2, {0, 2, 2}, {1, 1}); }
  }

  std::istringstream input(bytes);
  BinaryReader reader(input);
  REQUIRE_THROWS_AS(decode(reader), SerializationError);
}

TEST_CASE("Serialization - reader requires a seekable stream", "[serialization]") {
  NonSeekableBuffer buffer("RCLIBMDL");
  std::istream input(&buffer);
  REQUIRE_THROWS_MATCHES(BinaryReader(input), SerializationError, MessageMatches(ContainsSubstring("seekable")));
}

TEST_CASE("Serialization - reader starts at the current stream position", "[serialization]") {
  std::stringstream buffer;
  buffer << "junk";
  BinaryWriter(buffer).writeInt(7);
  buffer.seekg(4);

  BinaryReader reader(buffer);
  REQUIRE(reader.readInt() == 7);
  REQUIRE_THROWS_AS(reader.readU8(), SerializationError);
}

TEST_CASE("Serialization - writer reports stream failures", "[serialization]") {
  FailingBuffer buffer;
  std::ostream output(&buffer);
  SECTION("Stream without exceptions") {}
  SECTION("Stream with exceptions enabled") { output.exceptions(std::ios::badbit | std::ios::failbit); }

  BinaryWriter writer(output);
  REQUIRE_THROWS_MATCHES(writer.writeInt(1), SerializationError, MessageMatches(ContainsSubstring("failed to write")));
}

TEST_CASE("Serialization - writer rejects strings above the maximum length", "[serialization]") {
  std::stringstream buffer;
  BinaryWriter writer(buffer);
  REQUIRE_THROWS_AS(writer.writeString(std::string(serialization_max_string_length + 1, 'x')), SerializationError);
}

TEST_CASE("Serialization - translateSerializationErrors", "[serialization]") {
  REQUIRE(translateSerializationErrors("context", [] { return 42; }) == 42);
  REQUIRE_NOTHROW(translateSerializationErrors("context", [] {}));

  SECTION("Other exceptions become SerializationError with context") {
    REQUIRE_THROWS_MATCHES(
        translateSerializationErrors("loading NvarReservoir", []() -> int { throw std::invalid_argument("bad"); }),
        SerializationError, MessageMatches(Catch::Matchers::Equals("loading NvarReservoir: bad")));
    REQUIRE_THROWS_AS(translateSerializationErrors("context", [] { throw std::length_error("too long"); }),
                      SerializationError);
  }

  SECTION("SerializationError passes through unchanged") {
    REQUIRE_THROWS_MATCHES(translateSerializationErrors("context", [] { throw SerializationError("original"); }),
                           SerializationError, MessageMatches(Catch::Matchers::Equals("original")));
  }

  SECTION("std::bad_alloc passes through unchanged") {
    REQUIRE_THROWS_AS(translateSerializationErrors("context", [] { throw std::bad_alloc(); }), std::bad_alloc);
  }
}

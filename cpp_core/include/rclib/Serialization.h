#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

/// Raised when a model cannot be saved or a model file cannot be loaded.
class SerializationError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Version written by BinaryWriter::writeHeader; BinaryReader accepts versions 1..this.
/// Version 2 added RandomSparseReservoir's spectral_radius_method.
constexpr std::uint32_t serialization_format_version = 2;

/// Longest string a model file may contain. Strings only hold component type tags.
constexpr std::uint32_t serialization_max_string_length = 64;

/// Runs `body` and rethrows any exception other than SerializationError and
/// std::bad_alloc as a SerializationError prefixed with `context`. Every public
/// save/load entry point uses this so callers see a single error type.
template <typename Body> auto translateSerializationErrors(const char *context, Body &&body) -> decltype(body()) {
  try {
    return body();
  } catch (const SerializationError &) {
    throw;
  } catch (const std::bad_alloc &) {
    throw;
  } catch (const std::exception &e) {
    throw SerializationError(std::string(context) + ": " + e.what());
  }
}

/// Sums reservoir output widths into a readout feature count. Throws
/// SerializationError if the total, plus a bias column, would overflow the int
/// widths used by Model and the readouts.
std::int64_t sumFeatureWidths(const std::vector<std::int64_t> &widths);

/// Writes the primitives of the rclib model format (little-endian, fixed width).
class BinaryWriter {
public:
  explicit BinaryWriter(std::ostream &os);

  /// Writes the magic bytes and the current format version.
  void writeHeader();
  void writeBool(bool value);
  void writeU8(std::uint8_t value);
  void writeInt(std::int32_t value);
  void writeUInt(std::uint32_t value);
  void writeDouble(double value);
  /// u32 length followed by the bytes; at most serialization_max_string_length bytes.
  void writeString(const std::string &value);
  /// i64 rows, i64 cols, then the values in column-major order.
  void writeMatrix(const Eigen::MatrixXd &matrix);
  /// i64 rows, i64 cols, i64 nnz, then Eigen's compressed column layout:
  /// i32 outer[cols + 1], i32 inner[nnz], f64 values[nnz].
  void writeSparse(const Eigen::SparseMatrix<double> &matrix);

private:
  void writeInt64(std::int64_t value);
  void writeBytes(const void *data, std::size_t size);

  std::ostream &os;
};

/// Reads the primitives written by BinaryWriter. Every size is checked against
/// the bytes left in the input before anything is allocated, so malformed input
/// raises SerializationError instead of over-allocating or reading out of bounds.
class BinaryReader {
public:
  /// The stream must be seekable: the reader measures the bytes left in it.
  explicit BinaryReader(std::istream &is);

  /// Checks the magic bytes and rejects unknown format versions.
  void readHeader();
  /// The version read by readHeader. Before it is called, the current version: component
  /// payloads written without a header are in the current format. Loaders read fields added
  /// in a later version only when formatVersion() is at least that version.
  std::uint32_t formatVersion() const { return format_version; }
  /// Bytes left in the input after what has been read so far.
  std::uint64_t remainingBytes() const { return remaining; }
  bool readBool();
  std::uint8_t readU8();
  std::int32_t readInt();
  std::uint32_t readUInt();
  double readDouble();
  std::string readString();
  Eigen::MatrixXd readMatrix();
  Eigen::SparseMatrix<double> readSparse();

private:
  std::int64_t readInt64();
  /// Reads a matrix dimension or entry count and checks it lies in [0, INT32_MAX].
  std::int64_t readCount(const char *what);
  void readBytes(void *data, std::size_t size);

  std::istream &is;
  std::uint64_t remaining;
  std::uint32_t format_version = serialization_format_version;
};

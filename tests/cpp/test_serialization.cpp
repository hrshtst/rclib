#include "rclib/Serialization.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <catch2/catch_all.hpp>
#include <cstdint>
#include <cstring>
#include <functional>
#include <istream>
#include <limits>
#include <memory>
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
using Catch::Matchers::StartsWith;

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

// Serializes a component with its save() method.
template <typename Component> std::string saveToBytes(const Component &component) {
  std::stringstream buffer;
  BinaryWriter writer(buffer);
  component.save(writer);
  return buffer.str();
}

// Loads a component with its static load() method and checks that exactly the
// payload was consumed.
template <typename Component> std::shared_ptr<Component> loadFromBytes(const std::string &bytes) {
  std::istringstream input(bytes);
  BinaryReader reader(input);
  auto component = Component::load(reader);
  REQUIRE(reader.remainingBytes() == 0);
  return component;
}

// Ones on the main diagonal; also valid for rectangular shapes, unlike setIdentity().
Eigen::SparseMatrix<double> sparseIdentity(Eigen::Index rows, Eigen::Index cols) {
  Eigen::SparseMatrix<double> matrix(rows, cols);
  for (Eigen::Index i = 0; i < std::min(rows, cols); ++i) {
    matrix.insert(i, i) = 1.0;
  }
  matrix.makeCompressed();
  return matrix;
}

// A RandomSparseReservoir payload built field by field, for crafting invalid input.
struct RandomSparsePayload {
  std::int32_t n_neurons = 3;
  double spectral_radius = 0.9;
  double sparsity = 0.5;
  double leak_rate = 0.5;
  double input_scaling = 1.0;
  bool include_bias = false;
  std::uint32_t seed = 1;
  Eigen::SparseMatrix<double> W_res = sparseIdentity(3, 3);
  Eigen::MatrixXd bias = Eigen::MatrixXd::Zero(1, 3);
  Eigen::MatrixXd state = Eigen::MatrixXd::Zero(1, 3);
  bool w_in_initialized = true;
  Eigen::MatrixXd W_in = Eigen::MatrixXd::Ones(2, 3);

  std::string bytes() const {
    std::stringstream buffer;
    BinaryWriter writer(buffer);
    writer.writeInt(n_neurons);
    writer.writeDouble(spectral_radius);
    writer.writeDouble(sparsity);
    writer.writeDouble(leak_rate);
    writer.writeDouble(input_scaling);
    writer.writeBool(include_bias);
    writer.writeUInt(seed);
    writer.writeSparse(W_res);
    writer.writeMatrix(bias);
    writer.writeMatrix(state);
    writer.writeBool(w_in_initialized);
    if (w_in_initialized) {
      writer.writeMatrix(W_in);
    }
    return buffer.str();
  }
};

// An NvarReservoir payload built field by field, for crafting invalid input.
struct NvarPayload {
  std::int32_t num_lags = 2;
  std::int32_t polynomial_order = 1;
  bool initialized = true;
  std::int32_t input_dim = 1;
  Eigen::MatrixXd state = Eigen::MatrixXd::Zero(1, 2);
  Eigen::MatrixXd past_inputs = Eigen::MatrixXd::Zero(2, 1);

  std::string bytes() const {
    std::stringstream buffer;
    BinaryWriter writer(buffer);
    writer.writeInt(num_lags);
    writer.writeInt(polynomial_order);
    writer.writeBool(initialized);
    if (initialized) {
      writer.writeInt(input_dim);
      writer.writeMatrix(state);
      writer.writeMatrix(past_inputs);
    }
    return buffer.str();
  }
};

// Readout payloads built field by field, for crafting invalid input. Booleans and
// enum codes are raw bytes so that out-of-range encodings can be written.
struct RidgePayload {
  double alpha = 1e-3;
  std::uint8_t include_bias = 1;
  std::uint8_t solver = 1; // CHOLESKY
  double tolerance = 1e-6;
  std::uint8_t effective_solver = 1;
  std::uint8_t fitted = 1;
  Eigen::MatrixXd W_out = Eigen::MatrixXd::Ones(3, 2);

  std::string bytes() const {
    std::stringstream buffer;
    BinaryWriter writer(buffer);
    writer.writeDouble(alpha);
    writer.writeU8(include_bias);
    writer.writeU8(solver);
    writer.writeDouble(tolerance);
    writer.writeU8(effective_solver);
    writer.writeU8(fitted);
    if (fitted != 0) {
      writer.writeMatrix(W_out);
    }
    return buffer.str();
  }
};

struct RlsPayload {
  double lambda = 0.99;
  double delta = 1.0;
  std::uint8_t include_bias = 1;
  std::uint8_t solver = 0; // RANK1_UPDATE
  std::uint8_t initialized = 1;
  Eigen::MatrixXd W_out = Eigen::MatrixXd::Ones(3, 2);
  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(3, 3);

  std::string bytes() const {
    std::stringstream buffer;
    BinaryWriter writer(buffer);
    writer.writeDouble(lambda);
    writer.writeDouble(delta);
    writer.writeU8(include_bias);
    writer.writeU8(solver);
    writer.writeU8(initialized);
    if (initialized != 0) {
      writer.writeMatrix(W_out);
      writer.writeMatrix(P);
    }
    return buffer.str();
  }
};

struct LmsPayload {
  double learning_rate = 0.01;
  std::uint8_t include_bias = 1;
  std::uint8_t initialized = 1;
  Eigen::MatrixXd W_out = Eigen::MatrixXd::Ones(3, 2);

  std::string bytes() const {
    std::stringstream buffer;
    BinaryWriter writer(buffer);
    writer.writeDouble(learning_rate);
    writer.writeU8(include_bias);
    writer.writeU8(initialized);
    if (initialized != 0) {
      writer.writeMatrix(W_out);
    }
    return buffer.str();
  }
};

// Every strict prefix of `bytes` must be rejected by Component::load.
template <typename Component> void requireTruncationsRejected(const std::string &bytes) {
  for (std::size_t size = 0; size < bytes.size(); ++size) {
    std::istringstream input(bytes.substr(0, size));
    BinaryReader reader(input);
    REQUIRE_THROWS_AS(Component::load(reader), SerializationError);
  }
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

TEST_CASE("RandomSparseReservoir - serialization round-trips", "[serialization][RandomSparseReservoir]") {
  RandomSparseReservoir original(20, 0.9, 0.3, 0.5, 0.8, true, 123);
  const Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(6, 3);

  SECTION("Before the first input (W_in not generated yet)") {}
  SECTION("After advancing") {
    for (int i = 0; i < 3; ++i) {
      original.advance(inputs.row(i));
    }
  }

  const std::string bytes = saveToBytes(original);
  const auto restored = loadFromBytes<RandomSparseReservoir>(bytes);
  REQUIRE(saveToBytes(*restored) == bytes);

  REQUIRE(restored->getNNeurons() == original.getNNeurons());
  REQUIRE(restored->getSpectralRadius() == original.getSpectralRadius());
  REQUIRE(restored->getSparsity() == original.getSparsity());
  REQUIRE(restored->getLeakRate() == original.getLeakRate());
  REQUIRE(restored->getInputScaling() == original.getInputScaling());
  REQUIRE(restored->getIncludeBias() == original.getIncludeBias());
  REQUIRE(restored->getSeed() == original.getSeed());
  REQUIRE(restored->getInputDim() == original.getInputDim());
  REQUIRE(sameBits(restored->getState(), original.getState()));

  // Later updates are bit-identical.
  for (int i = 3; i < 6; ++i) {
    REQUIRE(sameBits(restored->advance(inputs.row(i)), original.advance(inputs.row(i))));
  }
}

TEST_CASE("NvarReservoir - serialization round-trips", "[serialization][NvarReservoir]") {
  NvarReservoir original(3, 2);
  const Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(6, 2);

  SECTION("Before the first input") {}
  SECTION("After advancing") {
    for (int i = 0; i < 3; ++i) {
      original.advance(inputs.row(i));
    }
  }

  const std::string bytes = saveToBytes(original);
  const auto restored = loadFromBytes<NvarReservoir>(bytes);
  REQUIRE(saveToBytes(*restored) == bytes);

  REQUIRE(restored->getNumLags() == original.getNumLags());
  REQUIRE(restored->getPolynomialOrder() == original.getPolynomialOrder());
  REQUIRE(restored->getInputDim() == original.getInputDim());
  REQUIRE(sameBits(restored->getState(), original.getState()));

  for (int i = 3; i < 6; ++i) {
    REQUIRE(sameBits(restored->advance(inputs.row(i)), original.advance(inputs.row(i))));
  }
}

TEST_CASE("RandomSparseReservoir - invalid payloads are rejected", "[serialization][RandomSparseReservoir]") {
  RandomSparsePayload payload;
  REQUIRE_NOTHROW(loadFromBytes<RandomSparseReservoir>(payload.bytes()));

  SECTION("Non-positive n_neurons") { payload.n_neurons = 0; }
  SECTION("NaN leak_rate") { payload.leak_rate = nan_value; }
  SECTION("Infinite spectral_radius") { payload.spectral_radius = inf_value; }
  SECTION("Non-square W_res") { payload.W_res = sparseIdentity(3, 4); }
  SECTION("W_res of the wrong size") { payload.W_res = sparseIdentity(4, 4); }
  SECTION("bias that is not a row vector") { payload.bias = Eigen::MatrixXd::Zero(3, 1); }
  SECTION("bias of the wrong length") { payload.bias = Eigen::MatrixXd::Zero(1, 2); }
  SECTION("state of the wrong width") { payload.state = Eigen::MatrixXd::Zero(1, 4); }
  SECTION("W_in with the wrong column count") { payload.W_in = Eigen::MatrixXd::Ones(2, 4); }
  SECTION("W_in without rows although initialized") { payload.W_in = Eigen::MatrixXd(0, 3); }

  REQUIRE_THROWS_MATCHES(loadFromBytes<RandomSparseReservoir>(payload.bytes()), SerializationError,
                         MessageMatches(StartsWith("RandomSparseReservoir: ")));
}

TEST_CASE("NvarReservoir - invalid payloads are rejected", "[serialization][NvarReservoir]") {
  NvarPayload payload;
  REQUIRE_NOTHROW(loadFromBytes<NvarReservoir>(payload.bytes()));

  SECTION("Non-positive num_lags") { payload.num_lags = 0; }
  SECTION("polynomial_order above the maximum") { payload.polynomial_order = 33; }
  SECTION("Non-positive input_dim although initialized") { payload.input_dim = 0; }
  SECTION("state of the wrong width") { payload.state = Eigen::MatrixXd::Zero(1, 3); }
  SECTION("past_inputs of the wrong shape") { payload.past_inputs = Eigen::MatrixXd::Zero(1, 1); }
  SECTION("Feature count above the supported maximum") {
    // getOutputDim throws std::length_error here; it must surface as SerializationError.
    payload.num_lags = 100;
    payload.polynomial_order = 4;
    payload.input_dim = 10;
  }

  REQUIRE_THROWS_MATCHES(loadFromBytes<NvarReservoir>(payload.bytes()), SerializationError,
                         MessageMatches(StartsWith("NvarReservoir: ")));
}

TEST_CASE("Reservoirs - truncated payloads are rejected", "[serialization]") {
  RandomSparseReservoir random_sparse(5, 0.9, 0.5, 0.5, 1.0, true);
  NvarReservoir nvar(2, 2);
  random_sparse.advance(Eigen::MatrixXd::Random(1, 2));
  nvar.advance(Eigen::MatrixXd::Random(1, 2));

  requireTruncationsRejected<RandomSparseReservoir>(saveToBytes(random_sparse));
  requireTruncationsRejected<NvarReservoir>(saveToBytes(nvar));
}

TEST_CASE("RidgeReadout - serialization round-trips", "[serialization][RidgeReadout]") {
  const auto solver = GENERATE(RidgeReadout::AUTO, RidgeReadout::CHOLESKY, RidgeReadout::DUAL_CHOLESKY,
                               RidgeReadout::CONJUGATE_GRADIENT, RidgeReadout::CONJUGATE_GRADIENT_IMPLICIT);
  const bool include_bias = GENERATE(true, false);
  RidgeReadout original(1e-3, include_bias, solver, 1e-9);
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(30, 6);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(30, 2);

  SECTION("Unfitted") {}
  SECTION("Fitted") { original.fit(states, targets); }

  const std::string bytes = saveToBytes(original);
  const auto restored = loadFromBytes<RidgeReadout>(bytes);
  REQUIRE(saveToBytes(*restored) == bytes);

  REQUIRE(restored->getAlpha() == original.getAlpha());
  REQUIRE(restored->getIncludeBias() == original.getIncludeBias());
  REQUIRE(restored->getSolver() == original.getSolver());
  REQUIRE(restored->getEffectiveSolver() == original.getEffectiveSolver());
  REQUIRE(restored->getTolerance() == original.getTolerance());
  REQUIRE(restored->getInputDim() == original.getInputDim());
  if (original.getInputDim() == 0) {
    REQUIRE_THROWS(restored->predict(states));
  } else {
    REQUIRE(sameBits(restored->getWeights(), original.getWeights()));
    REQUIRE(sameBits(restored->predict(states), original.predict(states)));
  }
}

TEST_CASE("RlsReadout - serialization round-trips and continues identically", "[serialization][RlsReadout]") {
  // The rank-k (Woodbury) update only runs for lambda == 1 and batches of more than one row.
  const auto config =
      GENERATE(std::make_pair(0.99, RlsReadout::RANK1_UPDATE), std::make_pair(1.0, RlsReadout::RANK_K_UPDATE));
  RlsReadout original(config.first, 0.5, true, config.second);
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(12, 4);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(12, 2);

  SECTION("Unfitted") {}
  SECTION("Fitted") { original.partialFit(states.topRows(4), targets.topRows(4)); }
  SECTION("Unfitted by a failed fit that left stale weights") {
    original.partialFit(states.topRows(4), targets.topRows(4));
    REQUIRE_THROWS(original.fit(Eigen::MatrixXd(0, 4), Eigen::MatrixXd(0, 2)));
  }

  const std::string bytes = saveToBytes(original);
  const auto restored = loadFromBytes<RlsReadout>(bytes);
  REQUIRE(saveToBytes(*restored) == bytes);
  REQUIRE(restored->getLambda() == original.getLambda());
  REQUIRE(restored->getDelta() == original.getDelta());
  REQUIRE(restored->getIncludeBias() == original.getIncludeBias());
  REQUIRE(restored->getSolver() == original.getSolver());
  REQUIRE(restored->getInputDim() == original.getInputDim());

  // The same later updates, in two-row batches, keep both readouts bit-identical.
  for (Eigen::Index start = 4; start < 12; start += 2) {
    original.partialFit(states.middleRows(start, 2), targets.middleRows(start, 2));
    restored->partialFit(states.middleRows(start, 2), targets.middleRows(start, 2));
  }
  REQUIRE(sameBits(restored->predict(states), original.predict(states)));
}

TEST_CASE("LmsReadout - serialization round-trips and continues identically", "[serialization][LmsReadout]") {
  LmsReadout original(0.05, true);
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(12, 4);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(12, 2);

  SECTION("Unfitted") {}
  SECTION("Fitted") { original.partialFit(states.topRows(4), targets.topRows(4)); }

  const std::string bytes = saveToBytes(original);
  const auto restored = loadFromBytes<LmsReadout>(bytes);
  REQUIRE(saveToBytes(*restored) == bytes);
  REQUIRE(restored->getLearningRate() == original.getLearningRate());
  REQUIRE(restored->getIncludeBias() == original.getIncludeBias());
  REQUIRE(restored->getInputDim() == original.getInputDim());

  for (Eigen::Index start = 4; start < 12; start += 2) {
    original.partialFit(states.middleRows(start, 2), targets.middleRows(start, 2));
    restored->partialFit(states.middleRows(start, 2), targets.middleRows(start, 2));
  }
  REQUIRE(sameBits(restored->predict(states), original.predict(states)));
}

TEST_CASE("RidgeReadout - invalid payloads are rejected", "[serialization][RidgeReadout]") {
  RidgePayload payload;
  REQUIRE_NOTHROW(loadFromBytes<RidgeReadout>(payload.bytes()));

  SECTION("NaN alpha") { payload.alpha = nan_value; }
  SECTION("Infinite tolerance") { payload.tolerance = inf_value; }
  SECTION("include_bias byte other than 0 or 1") { payload.include_bias = 2; }
  SECTION("Unknown solver code") { payload.solver = 5; }
  SECTION("Unknown effective solver code") { payload.effective_solver = 99; }
  SECTION("Effective solver differs from an explicit solver") { payload.effective_solver = 2; }
  SECTION("Fitted without weights") { payload.W_out = Eigen::MatrixXd(0, 0); }
  SECTION("Weights without room for the bias row") { payload.W_out = Eigen::MatrixXd::Ones(1, 2); }

  REQUIRE_THROWS_AS(loadFromBytes<RidgeReadout>(payload.bytes()), SerializationError);
}

TEST_CASE("RlsReadout - invalid payloads are rejected", "[serialization][RlsReadout]") {
  RlsPayload payload;
  REQUIRE_NOTHROW(loadFromBytes<RlsReadout>(payload.bytes()));

  SECTION("NaN lambda") { payload.lambda = nan_value; }
  SECTION("Non-positive delta") { payload.delta = 0.0; }
  SECTION("Unknown solver code") { payload.solver = 2; }
  SECTION("initialized byte other than 0 or 1") { payload.initialized = 2; }
  SECTION("Weights without output columns") { payload.W_out = Eigen::MatrixXd(3, 0); }
  SECTION("Non-square P") { payload.P = Eigen::MatrixXd::Identity(3, 2); }
  SECTION("P of the wrong size") { payload.P = Eigen::MatrixXd::Identity(2, 2); }

  REQUIRE_THROWS_AS(loadFromBytes<RlsReadout>(payload.bytes()), SerializationError);
}

TEST_CASE("LmsReadout - invalid payloads are rejected", "[serialization][LmsReadout]") {
  LmsPayload payload;
  REQUIRE_NOTHROW(loadFromBytes<LmsReadout>(payload.bytes()));

  SECTION("NaN learning_rate") { payload.learning_rate = nan_value; }
  SECTION("Weights without room for the bias row") { payload.W_out = Eigen::MatrixXd::Ones(1, 2); }
  SECTION("Weights without output columns") { payload.W_out = Eigen::MatrixXd(3, 0); }

  REQUIRE_THROWS_AS(loadFromBytes<LmsReadout>(payload.bytes()), SerializationError);
}

TEST_CASE("Readouts - truncated payloads are rejected", "[serialization]") {
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(6, 3);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(6, 2);
  RidgeReadout ridge(1e-3, true);
  RlsReadout rls(0.99, 1.0, true);
  LmsReadout lms(0.01, true);
  ridge.fit(states, targets);
  rls.fit(states, targets);
  lms.fit(states, targets);

  requireTruncationsRejected<RidgeReadout>(saveToBytes(ridge));
  requireTruncationsRejected<RlsReadout>(saveToBytes(rls));
  requireTruncationsRejected<LmsReadout>(saveToBytes(lms));
}

#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;

namespace {

bool sameBits(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         (a.size() == 0 || std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(double)) == 0);
}

std::string saveToBytes(const Model &model) {
  std::ostringstream buffer;
  model.save(buffer);
  return buffer.str();
}

Model loadFromBytes(const std::string &bytes) {
  std::istringstream input(bytes);
  return Model::load(input);
}

enum class Topology { RandomSparse, Nvar, SerialMixed, ParallelMixed };
enum class ReadoutKind { Ridge, RlsRank1, RlsRankK, Lms };

std::shared_ptr<Readout> makeReadout(ReadoutKind kind) {
  switch (kind) {
    case ReadoutKind::Ridge:
      return std::make_shared<RidgeReadout>(1e-4, true);
    case ReadoutKind::RlsRank1:
      return std::make_shared<RlsReadout>(0.99, 1.0, true, RlsReadout::RANK1_UPDATE);
    case ReadoutKind::RlsRankK:
      // The rank-k (Woodbury) update only runs for lambda == 1 and batches of more than one row.
      return std::make_shared<RlsReadout>(1.0, 1.0, true, RlsReadout::RANK_K_UPDATE);
    case ReadoutKind::Lms:
      return std::make_shared<LmsReadout>(0.01, true);
  }
  return nullptr;
}

Model makeModel(Topology topology, ReadoutKind readout) {
  Model model;
  switch (topology) {
    case Topology::RandomSparse:
      model.addReservoir(std::make_shared<RandomSparseReservoir>(20, 0.9, 0.3, 0.5, 1.0, true, 7));
      break;
    case Topology::Nvar:
      model.addReservoir(std::make_shared<NvarReservoir>(3, 2));
      break;
    case Topology::SerialMixed:
      model.addReservoir(std::make_shared<RandomSparseReservoir>(6, 0.9, 0.5, 0.5, 1.0, true, 7));
      model.addReservoir(std::make_shared<NvarReservoir>(2, 2));
      break;
    case Topology::ParallelMixed:
      model.addReservoir(std::make_shared<RandomSparseReservoir>(20, 0.9, 0.3, 0.5, 1.0, true, 7), "parallel");
      model.addReservoir(std::make_shared<NvarReservoir>(2, 2), "parallel");
      break;
  }
  model.setReadout(makeReadout(readout));
  return model;
}

// A one-column signal whose next value is the target, so generative prediction applies.
Eigen::MatrixXd signal(int rows, double phase) {
  Eigen::MatrixXd values(rows, 1);
  for (int i = 0; i < rows; ++i) {
    values(i, 0) = std::sin(0.3 * i + phase);
  }
  return values;
}

// Writes a header followed by whatever `body` writes, for crafting model files.
std::string craftModel(const std::function<void(BinaryWriter &)> &body) {
  std::ostringstream buffer;
  BinaryWriter writer(buffer);
  writer.writeHeader();
  body(writer);
  return buffer.str();
}

class MinimalReservoir : public Reservoir {
public:
  const Eigen::MatrixXd &advance(const Eigen::MatrixXd & /*input*/) override { return state; }
  void resetState() override { state.setZero(); }
  const Eigen::MatrixXd &getState() const override { return state; }

private:
  Eigen::MatrixXd state = Eigen::MatrixXd::Zero(1, 3);
};

} // namespace

TEST_CASE("Model serialization - round trips predict bit-identically", "[serialization][Model]") {
  const auto topology =
      GENERATE(Topology::RandomSparse, Topology::Nvar, Topology::SerialMixed, Topology::ParallelMixed);
  const auto readout = GENERATE(ReadoutKind::Ridge, ReadoutKind::RlsRank1, ReadoutKind::RlsRankK, ReadoutKind::Lms);
  Model original = makeModel(topology, readout);
  const Eigen::MatrixXd series = signal(61, 0.0);
  original.fit(series.topRows(60), series.bottomRows(60), 5);

  const std::string bytes = saveToBytes(original);
  Model restored = loadFromBytes(bytes);
  REQUIRE(saveToBytes(restored) == bytes);
  REQUIRE(restored.getNumReservoirs() == original.getNumReservoirs());
  REQUIRE(restored.getConnectionType() == original.getConnectionType());

  const Eigen::MatrixXd probe = signal(15, 1.0);
  REQUIRE(sameBits(restored.predict(probe), original.predict(probe)));
}

TEST_CASE("Model serialization - online learning continues identically", "[serialization][Model]") {
  const auto topology = GENERATE(Topology::RandomSparse, Topology::SerialMixed, Topology::ParallelMixed);
  const auto readout = GENERATE(ReadoutKind::RlsRank1, ReadoutKind::RlsRankK, ReadoutKind::Lms);
  Model original = makeModel(topology, readout);
  const Eigen::MatrixXd series = signal(41, 0.0);
  original.fit(series.topRows(20), series.middleRows(1, 20));

  Model restored = loadFromBytes(saveToBytes(original));
  // The saved reservoir states carry over, so predictOnline continues from the same point.
  for (int start = 20; start < 40; start += 2) {
    const Eigen::MatrixXd input = series.middleRows(start, 2);
    const Eigen::MatrixXd target = series.middleRows(start + 1, 2);
    REQUIRE(sameBits(restored.predictOnline(input), original.predictOnline(input)));
    restored.partialFit(input, target);
    original.partialFit(input, target);
  }
  REQUIRE(sameBits(restored.predict(series), original.predict(series)));
}

TEST_CASE("Model serialization - generative prediction continues identically", "[serialization][Model]") {
  const auto topology =
      GENERATE(Topology::RandomSparse, Topology::Nvar, Topology::SerialMixed, Topology::ParallelMixed);
  Model original = makeModel(topology, ReadoutKind::Ridge);
  const Eigen::MatrixXd series = signal(81, 0.0);
  original.fit(series.topRows(80), series.bottomRows(80), 10);

  Model restored = loadFromBytes(saveToBytes(original));
  // Without priming input, generation starts from the saved reservoir states.
  const Eigen::MatrixXd no_priming(0, 1);
  REQUIRE(sameBits(restored.predictGenerative(no_priming, 10), original.predictGenerative(no_priming, 10)));
}

TEST_CASE("Model serialization - unfitted and uninitialized components round-trip", "[serialization][Model]") {
  const auto readout = GENERATE(ReadoutKind::Ridge, ReadoutKind::RlsRank1, ReadoutKind::Lms);
  Model original = makeModel(Topology::ParallelMixed, readout);

  const std::string bytes = saveToBytes(original);
  Model restored = loadFromBytes(bytes);
  REQUIRE(saveToBytes(restored) == bytes);
  REQUIRE(restored.getReservoir(0)->getInputDim() == 0);
  REQUIRE(restored.getReadout()->getInputDim() == 0);

  const Eigen::MatrixXd probe = signal(5, 0.0);
  REQUIRE_THROWS(restored.predict(probe));
  REQUIRE_THROWS(original.predict(probe));
}

TEST_CASE("Model serialization - a readout left unfitted by a failed fit", "[serialization][Model]") {
  const auto readout = GENERATE(ReadoutKind::RlsRank1, ReadoutKind::Lms);
  Model original = makeModel(Topology::RandomSparse, readout);
  const Eigen::MatrixXd series = signal(41, 0.0);
  original.fit(series.topRows(20), series.middleRows(1, 20));

  // RLS throws on an empty batch and LMS accepts it; both end up unfitted with
  // their previous weights still allocated.
  try {
    original.getReadout()->fit(Eigen::MatrixXd(0, 20), Eigen::MatrixXd(0, 1));
  } catch (const std::invalid_argument &) {
  }
  REQUIRE(original.getReadout()->getInputDim() == 0);

  Model restored = loadFromBytes(saveToBytes(original));
  REQUIRE(restored.getReadout()->getInputDim() == 0);
  // predict() advances the reservoirs before the readout throws, so call it on both.
  REQUIRE_THROWS(restored.predict(series));
  REQUIRE_THROWS(original.predict(series));

  // The same later updates restart both readouts identically.
  for (int start = 20; start < 40; start += 2) {
    original.partialFit(series.middleRows(start, 2), series.middleRows(start + 1, 2));
    restored.partialFit(series.middleRows(start, 2), series.middleRows(start + 1, 2));
  }
  REQUIRE(sameBits(restored.predict(series), original.predict(series)));
}

TEST_CASE("Model serialization - models that cannot be saved", "[serialization][Model]") {
  Model model;
  const Eigen::MatrixXd series = signal(21, 0.0);
  std::string expected_message;

  SECTION("No reservoir") {
    model.setReadout(std::make_shared<RidgeReadout>());
    expected_message = "at least one reservoir and a readout";
  }
  SECTION("No readout") {
    model.addReservoir(std::make_shared<NvarReservoir>(2));
    expected_message = "at least one reservoir and a readout";
  }
  SECTION("The same reservoir object twice") {
    auto reservoir = std::make_shared<RandomSparseReservoir>(5, 0.9);
    model.addReservoir(reservoir);
    model.addReservoir(reservoir);
    model.setReadout(std::make_shared<RidgeReadout>());
    expected_message = "same reservoir object";
  }
  SECTION("A reservoir type other than the built-in ones") {
    model.addReservoir(std::make_shared<MinimalReservoir>());
    model.setReadout(std::make_shared<RidgeReadout>());
    expected_message = "unsupported reservoir type";
  }
  SECTION("An independently fitted readout of the wrong width") {
    auto reservoir = std::make_shared<RandomSparseReservoir>(5, 0.9);
    reservoir->advance(series.topRows(1)); // locks the input width
    model.addReservoir(reservoir);
    auto readout = std::make_shared<RidgeReadout>();
    readout->fit(Eigen::MatrixXd::Random(10, 7), Eigen::MatrixXd::Random(10, 1));
    model.setReadout(readout);
    expected_message = "fitted to 7 features but the reservoirs output 5";
  }

  std::ostringstream output;
  REQUIRE_THROWS_MATCHES(model.save(output), SerializationError, MessageMatches(ContainsSubstring(expected_message)));
  REQUIRE(output.str().empty()); // validation runs before anything is written
}

TEST_CASE("Model serialization - invalid model files are rejected", "[serialization][Model]") {
  // Components whose input widths are locked, used to build inconsistent models.
  auto locked_random_sparse = [](int n_neurons, int input_width) {
    auto reservoir = std::make_shared<RandomSparseReservoir>(n_neurons, 0.9, 0.5, 0.5, 1.0);
    reservoir->advance(Eigen::MatrixXd::Ones(1, input_width));
    return reservoir;
  };
  RidgeReadout unfitted_ridge;
  RidgeReadout ridge_7;
  ridge_7.fit(Eigen::MatrixXd::Random(10, 7), Eigen::MatrixXd::Random(10, 1));
  const auto write_ridge = [&](BinaryWriter &writer, const RidgeReadout &readout) {
    writer.writeString("RidgeReadout");
    readout.save(writer);
  };
  const auto write_random_sparse = [](BinaryWriter &writer, const RandomSparseReservoir &reservoir) {
    writer.writeString("RandomSparseReservoir");
    reservoir.save(writer);
  };

  std::string bytes;
  std::string expected_message;

  SECTION("Unknown connection code") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(7);
      writer.writeUInt(1);
      write_random_sparse(writer, *locked_random_sparse(5, 1));
      write_ridge(writer, unfitted_ridge);
    });
    expected_message = "unknown connection code";
  }
  SECTION("No reservoirs") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(0);
      write_ridge(writer, unfitted_ridge);
    });
    expected_message = "at least one reservoir";
  }
  SECTION("A huge reservoir count without the data behind it") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(std::numeric_limits<std::uint32_t>::max());
    });
    expected_message = "end of model data";
  }
  SECTION("Unknown reservoir type tag") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(1);
      writer.writeString("EchoReservoir");
    });
    expected_message = "unknown reservoir type tag";
  }
  SECTION("Unknown readout type tag") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(1);
      write_random_sparse(writer, *locked_random_sparse(5, 1));
      writer.writeString("KernelReadout");
    });
    expected_message = "unknown readout type tag";
  }
  SECTION("Serial reservoir locked to a different width than its predecessor outputs") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(2);
      write_random_sparse(writer, *locked_random_sparse(5, 1));
      write_random_sparse(writer, *locked_random_sparse(4, 3));
      write_ridge(writer, unfitted_ridge);
    });
    expected_message = "locked to an input width of 3 but the previous reservoir outputs 5";
  }
  SECTION("Parallel reservoirs locked to different input widths") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(1);
      writer.writeUInt(2);
      write_random_sparse(writer, *locked_random_sparse(5, 1));
      write_random_sparse(writer, *locked_random_sparse(4, 2));
      write_ridge(writer, unfitted_ridge);
    });
    expected_message = "different input widths";
  }
  SECTION("Readout fitted to a different width than the reservoirs output") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(1);
      writer.writeUInt(2);
      write_random_sparse(writer, *locked_random_sparse(5, 1));
      write_random_sparse(writer, *locked_random_sparse(4, 1));
      write_ridge(writer, ridge_7);
    });
    expected_message = "fitted to 7 features but the reservoirs output 9";
  }
  SECTION("Trailing data after the model") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(1);
      write_random_sparse(writer, *locked_random_sparse(5, 1));
      write_ridge(writer, unfitted_ridge);
      writer.writeU8(0);
    });
    expected_message = "unexpected data after the end of the model";
  }

  REQUIRE_THROWS_MATCHES(loadFromBytes(bytes), SerializationError, MessageMatches(ContainsSubstring(expected_message)));
}

TEST_CASE("Model serialization - every truncated file is rejected", "[serialization][Model]") {
  Model small;
  small.addReservoir(std::make_shared<RandomSparseReservoir>(4, 0.9, 0.5, 0.5, 1.0, true), "parallel");
  small.addReservoir(std::make_shared<NvarReservoir>(2, 1), "parallel");
  small.setReadout(std::make_shared<RlsReadout>(0.99, 1.0, true));
  const Eigen::MatrixXd series = signal(9, 0.0);
  small.fit(series.topRows(8), series.bottomRows(8));

  const std::string bytes = saveToBytes(small);
  for (std::size_t size = 0; size < bytes.size(); ++size) {
    REQUIRE_THROWS_AS(loadFromBytes(bytes.substr(0, size)), SerializationError);
  }
  REQUIRE_NOTHROW(loadFromBytes(bytes));
}

TEST_CASE("Model serialization - feature width sums are overflow-checked", "[serialization]") {
  constexpr std::int64_t int32_max = std::numeric_limits<std::int32_t>::max();
  REQUIRE(sumFeatureWidths({}) == 0);
  REQUIRE(sumFeatureWidths({5, 7}) == 12);
  // One column is reserved for the readout bias.
  REQUIRE(sumFeatureWidths({int32_max - 1}) == int32_max - 1);
  REQUIRE_THROWS_AS(sumFeatureWidths({int32_max}), SerializationError);
  REQUIRE_THROWS_AS(sumFeatureWidths({int32_max / 2 + 1, int32_max / 2 + 1}), SerializationError);
}

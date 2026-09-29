#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <algorithm>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#  include <csignal>
#  include <sys/resource.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  define RCLIB_TEST_HAS_FORK 1
#endif

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

// A fresh directory under the system temporary directory, removed with its contents.
class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::random_device random;
    path = std::filesystem::temp_directory_path() /
           ("rclib_test_" + std::to_string(random()) + "_" + std::to_string(random()));
    std::filesystem::create_directory(path);
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  std::filesystem::path path;
};

std::string readFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

// Sorted file names in a directory, to check that no temporary file is left behind.
std::vector<std::string> directoryEntries(const std::filesystem::path &directory) {
  std::vector<std::string> names;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    names.push_back(entry.path().filename().string());
  }
  std::sort(names.begin(), names.end());
  return names;
}

Model makeFittedModel(Topology topology) {
  Model model = makeModel(topology, ReadoutKind::Ridge);
  const Eigen::MatrixXd series = signal(41, 0.0);
  model.fit(series.topRows(40), series.bottomRows(40), 5);
  return model;
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

TEST_CASE("Model serialization - unused reservoirs with a matching fitted readout", "[serialization][Model]") {
  // Output widths known before any input (RandomSparse) must match the readout, and
  // a matching model saves, loads and predicts like the original.
  Model original;
  SECTION("Serial") { original.addReservoir(std::make_shared<RandomSparseReservoir>(5, 0.9, 0.5, 0.5, 1.0, true)); }
  SECTION("Parallel") {
    original.addReservoir(std::make_shared<RandomSparseReservoir>(5, 0.9, 0.5, 0.5, 1.0, true), "parallel");
    original.addReservoir(std::make_shared<RandomSparseReservoir>(4, 0.9, 0.5, 0.5, 1.0, true, 3), "parallel");
  }
  auto readout = std::make_shared<RidgeReadout>();
  const auto features = static_cast<Eigen::Index>(original.getNumReservoirs() == 1 ? 5 : 9);
  readout->fit(Eigen::MatrixXd::Random(20, features), Eigen::MatrixXd::Random(20, 1));
  original.setReadout(readout);

  Model restored = loadFromBytes(saveToBytes(original));
  const Eigen::MatrixXd probe = signal(6, 0.0);
  REQUIRE(sameBits(restored.predict(probe), original.predict(probe)));
}

TEST_CASE("Model serialization - a readout left unfitted by a failed fit", "[serialization][Model]") {
  Model original = makeModel(Topology::RandomSparse, ReadoutKind::RlsRank1);
  const Eigen::MatrixXd series = signal(41, 0.0);
  original.fit(series.topRows(20), series.middleRows(1, 20));

  // RLS clears its fitted flag before validating the batch, so a rejected fit
  // leaves it unfitted with its previous weights still allocated.
  REQUIRE_THROWS_AS(original.getReadout()->fit(Eigen::MatrixXd(0, 20), Eigen::MatrixXd(0, 1)), std::invalid_argument);
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
  // A RandomSparse reservoir's output width is known before it sees any input.
  SECTION("A readout of the wrong width after a reservoir that never saw input") {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(5, 0.9));
    auto readout = std::make_shared<RidgeReadout>();
    readout->fit(Eigen::MatrixXd::Random(10, 7), Eigen::MatrixXd::Random(10, 1));
    model.setReadout(readout);
    expected_message = "fitted to 7 features but the reservoirs output 5";
  }
  SECTION("A reservoir locked to a width that an unused predecessor does not output") {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(5, 0.9));
    auto nvar = std::make_shared<NvarReservoir>(2);
    nvar->advance(Eigen::MatrixXd::Ones(1, 7));
    model.addReservoir(nvar);
    model.setReadout(std::make_shared<RidgeReadout>());
    expected_message = "locked to an input width of 7 but the previous reservoir outputs 5";
  }
  SECTION("Unused parallel reservoirs and a readout of the wrong width") {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(5, 0.9), "parallel");
    model.addReservoir(std::make_shared<RandomSparseReservoir>(4, 0.9), "parallel");
    auto readout = std::make_shared<RidgeReadout>();
    readout->fit(Eigen::MatrixXd::Random(10, 7), Eigen::MatrixXd::Random(10, 1));
    model.setReadout(readout);
    expected_message = "fitted to 7 features but the reservoirs output 9";
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
  SECTION("Readout of a different width after an unused reservoir") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(1);
      write_random_sparse(writer, RandomSparseReservoir(5, 0.9));
      write_ridge(writer, ridge_7);
    });
    expected_message = "fitted to 7 features but the reservoirs output 5";
  }
  SECTION("Reservoir locked to a width that an unused predecessor does not output") {
    NvarReservoir nvar(2);
    nvar.advance(Eigen::MatrixXd::Ones(1, 7));
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(2);
      write_random_sparse(writer, RandomSparseReservoir(5, 0.9));
      writer.writeString("NvarReservoir");
      nvar.save(writer);
      write_ridge(writer, unfitted_ridge);
    });
    expected_message = "locked to an input width of 7 but the previous reservoir outputs 5";
  }
  SECTION("Unused parallel reservoirs and a readout of a different width") {
    bytes = craftModel([&](BinaryWriter &writer) {
      writer.writeU8(1);
      writer.writeUInt(2);
      write_random_sparse(writer, RandomSparseReservoir(5, 0.9));
      write_random_sparse(writer, RandomSparseReservoir(4, 0.9));
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

TEST_CASE("Model serialization - files round-trip and are replaced", "[serialization][Model]") {
  TemporaryDirectory directory;
  const std::string path = (directory.path / "model.rclib").string();

  const Model first = makeFittedModel(Topology::SerialMixed);
  first.save(path);
  REQUIRE(readFile(path) == saveToBytes(first));
  const Eigen::MatrixXd probe = signal(15, 1.0);
  Model restored = Model::load(path);
  Model first_copy = loadFromBytes(saveToBytes(first));
  REQUIRE(sameBits(restored.predict(probe), first_copy.predict(probe)));

  // Saving again replaces the file in place.
  const Model second = makeFittedModel(Topology::ParallelMixed);
  second.save(path);
  REQUIRE(readFile(path) == saveToBytes(second));
  REQUIRE(directoryEntries(directory.path) == std::vector<std::string>{"model.rclib"});
}

TEST_CASE("Model serialization - file errors", "[serialization][Model]") {
  TemporaryDirectory directory;

  SECTION("Loading a missing file") {
    REQUIRE_THROWS_MATCHES(Model::load((directory.path / "missing.rclib").string()), SerializationError,
                           MessageMatches(ContainsSubstring("cannot open")));
  }
  SECTION("Loading a truncated file") {
    const std::string bytes = saveToBytes(makeFittedModel(Topology::RandomSparse));
    const auto path = directory.path / "truncated.rclib";
    std::ofstream(path, std::ios::binary) << bytes.substr(0, bytes.size() / 2);
    REQUIRE_THROWS_AS(Model::load(path.string()), SerializationError);
  }
  SECTION("Saving into a missing directory") {
    const auto path = directory.path / "missing" / "model.rclib";
    REQUIRE_THROWS_MATCHES(makeFittedModel(Topology::RandomSparse).save(path.string()), SerializationError,
                           MessageMatches(ContainsSubstring("cannot create")));
    REQUIRE(directoryEntries(directory.path).empty());
  }
}

TEST_CASE("Model serialization - a rejected save keeps the existing file", "[serialization][Model]") {
  TemporaryDirectory directory;
  const std::string path = (directory.path / "model.rclib").string();
  makeFittedModel(Topology::RandomSparse).save(path);
  const std::string checkpoint = readFile(path);

  Model model;
  SECTION("No readout") { model.addReservoir(std::make_shared<NvarReservoir>(2)); }
  SECTION("The same reservoir object twice") {
    auto reservoir = std::make_shared<RandomSparseReservoir>(5, 0.9);
    model.addReservoir(reservoir);
    model.addReservoir(reservoir);
    model.setReadout(std::make_shared<RidgeReadout>());
  }
  SECTION("A reservoir type other than the built-in ones") {
    model.addReservoir(std::make_shared<MinimalReservoir>());
    model.setReadout(std::make_shared<RidgeReadout>());
  }
  SECTION("An independently fitted readout of the wrong width") {
    auto reservoir = std::make_shared<RandomSparseReservoir>(5, 0.9);
    reservoir->advance(Eigen::MatrixXd::Ones(1, 1));
    model.addReservoir(reservoir);
    auto readout = std::make_shared<RidgeReadout>();
    readout->fit(Eigen::MatrixXd::Random(10, 7), Eigen::MatrixXd::Random(10, 1));
    model.setReadout(readout);
  }
  SECTION("A readout of the wrong width after a reservoir that never saw input") {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(5, 0.9));
    auto readout = std::make_shared<RidgeReadout>();
    readout->fit(Eigen::MatrixXd::Random(10, 7), Eigen::MatrixXd::Random(10, 1));
    model.setReadout(readout);
  }

  REQUIRE_THROWS_AS(model.save(path), SerializationError);
  REQUIRE(readFile(path) == checkpoint);
  REQUIRE(directoryEntries(directory.path) == std::vector<std::string>{"model.rclib"});
}

TEST_CASE("Model serialization - a failed rename keeps the destination", "[serialization][Model]") {
  // The destination is a non-empty directory, so the final rename fails after the
  // temporary file has been written; the temporary file must be cleaned up.
  TemporaryDirectory directory;
  const auto destination = directory.path / "model.rclib";
  std::filesystem::create_directory(destination);
  std::ofstream(destination / "keep.txt") << "keep";

  REQUIRE_THROWS_MATCHES(makeFittedModel(Topology::RandomSparse).save(destination.string()), SerializationError,
                         MessageMatches(ContainsSubstring("cannot replace")));
  REQUIRE(std::filesystem::is_directory(destination));
  REQUIRE(directoryEntries(destination) == std::vector<std::string>{"keep.txt"});
  REQUIRE(directoryEntries(directory.path) == std::vector<std::string>{"model.rclib"});
}

#ifdef RCLIB_TEST_HAS_FORK
TEST_CASE("Model serialization - a failed write keeps the existing file", "[serialization][Model]") {
  TemporaryDirectory directory;
  const std::string path = (directory.path / "model.rclib").string();
  makeFittedModel(Topology::RandomSparse).save(path);
  const std::string checkpoint = readFile(path);
  const Model replacement = makeFittedModel(Topology::ParallelMixed);

  // The file-size limit and the SIGXFSZ disposition are process-wide, so they are
  // only changed in a child process, which reports through its exit code.
  const pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    int code = 3;                  // could not set the limit
    std::signal(SIGXFSZ, SIG_IGN); // writes past the limit then fail with EFBIG
    rlimit limit{};
    if (getrlimit(RLIMIT_FSIZE, &limit) == 0) {
      limit.rlim_cur = 64;
      if (setrlimit(RLIMIT_FSIZE, &limit) == 0) {
        try {
          replacement.save(path);
          code = 1; // the save unexpectedly succeeded
        } catch (const SerializationError &) {
          code = 0;
        } catch (...) {
          code = 2; // wrong exception type
        }
      }
    }
    _exit(code);
  }

  int status = 0;
  REQUIRE(waitpid(child, &status, 0) == child);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
  REQUIRE(readFile(path) == checkpoint);
  REQUIRE(directoryEntries(directory.path) == std::vector<std::string>{"model.rclib"});
}
#endif

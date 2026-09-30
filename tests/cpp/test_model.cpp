#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <algorithm>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("Model - configuration", "[Model]") {
  Model model;

  SECTION("Throws when not configured") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_THROWS(model.fit(inputs, targets));
    REQUIRE_THROWS_AS(model.fitSequences({inputs}, {targets}), std::runtime_error);
    REQUIRE_THROWS(model.predict(inputs));
    REQUIRE_THROWS(model.predictOnline(inputs.row(0)));
  }

  auto res = std::make_shared<RandomSparseReservoir>(10, 0.9, 0.5, 0.1, 1.0);
  model.addReservoir(res);

  SECTION("Throws when readout is not set") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_THROWS(model.fit(inputs, targets));
    REQUIRE_THROWS_AS(model.fitSequences({inputs}, {targets}), std::runtime_error);
    REQUIRE_THROWS(model.predict(inputs));
    REQUIRE_THROWS(model.predictOnline(inputs.row(0)));
  }

  auto readout = std::make_shared<RidgeReadout>();
  model.setReadout(readout);

  SECTION("Does not throw when fully configured") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_NOTHROW(model.fit(inputs, targets));
    REQUIRE_NOTHROW(model.fitSequences({inputs, inputs}, {targets, targets}));
    REQUIRE_NOTHROW(model.predict(inputs));
    REQUIRE_NOTHROW(model.predictOnline(inputs.row(0)));
  }
}

TEST_CASE("Model - fit and predict", "[Model]") {
  Model model;
  auto res = std::make_shared<RandomSparseReservoir>(100, 0.9, 0.1, 0.2, 1.0);
  auto readout = std::make_shared<RidgeReadout>(1e-6);
  model.addReservoir(res);
  model.setReadout(readout);

  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(200, 1);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(200, 1);

  model.fit(inputs, targets);
  Eigen::MatrixXd predictions = model.predict(inputs);

  REQUIRE(predictions.rows() == 200);
  REQUIRE(predictions.cols() == 1);

  double prediction_error = (predictions - targets).squaredNorm();
  double original_error = targets.squaredNorm();
  REQUIRE(prediction_error < original_error);
}

TEST_CASE("Model - parallel connection", "[Model]") {
  Model model;
  auto res1 = std::make_shared<RandomSparseReservoir>(50, 0.9, 0.1, 0.2, 1.0);
  auto res2 = std::make_shared<RandomSparseReservoir>(50, 0.9, 0.1, 0.2, 1.0);
  auto readout = std::make_shared<RidgeReadout>(1e-6);
  model.addReservoir(res1, "parallel");
  model.addReservoir(res2, "parallel");
  model.setReadout(readout);

  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(200, 1);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(200, 1);

  model.fit(inputs, targets);
  Eigen::MatrixXd predictions = model.predict(inputs);

  REQUIRE(predictions.rows() == 200);
  REQUIRE(predictions.cols() == 1);

  double prediction_error = (predictions - targets).squaredNorm();
  double original_error = targets.squaredNorm();
  REQUIRE(prediction_error < original_error);
}

TEST_CASE("Model - parallel reservoir errors propagate to the caller", "[Model]") {
  // Reservoir updates in a parallel model run inside an OpenMP region; an error
  // raised there must reach the caller as an exception instead of terminating.
  // Both reservoir types lock their input width on first use.
  Model model;
  SECTION("NvarReservoir") {
    model.addReservoir(std::make_shared<NvarReservoir>(2), "parallel");
    model.addReservoir(std::make_shared<NvarReservoir>(3), "parallel");
  }
  SECTION("RandomSparseReservoir") {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(10, 0.9, 0.5, 0.1, 1.0), "parallel");
    model.addReservoir(std::make_shared<RandomSparseReservoir>(8, 0.9, 0.5, 0.1, 1.0), "parallel");
  }
  model.setReadout(std::make_shared<RidgeReadout>(1e-6));

  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(20, 1);
  model.fit(inputs, inputs);

  Eigen::MatrixXd wider = Eigen::MatrixXd::Random(5, 2);
  REQUIRE_THROWS_AS(model.predict(wider), std::invalid_argument);
  REQUIRE_THROWS_AS(model.predictOnline(wider.row(0)), std::invalid_argument);

  // The model remains usable with the original input width.
  REQUIRE(model.predict(inputs).rows() == inputs.rows());
}

TEST_CASE("Model - resetReservoirs", "[Model]") {
  Model model;
  auto res1 = std::make_shared<RandomSparseReservoir>(10, 0.9, 0.1, 0.2, 1.0);
  auto res2 = std::make_shared<RandomSparseReservoir>(5, 0.8, 0.2, 0.3, 1.0);
  model.addReservoir(res1);
  model.addReservoir(res2);

  // Advance states to ensure they are not zero
  Eigen::MatrixXd input = Eigen::MatrixXd::Ones(1, 1);
  res1->advance(input);
  res2->advance(input);

  REQUIRE(res1->getState().norm() > 0);
  REQUIRE(res2->getState().norm() > 0);

  model.resetReservoirs();

  REQUIRE(res1->getState().norm() == 0);
  REQUIRE(res2->getState().norm() == 0);
}

TEST_CASE("Model - partialFit", "[Model]") {
  Model model;
  auto res = std::make_shared<RandomSparseReservoir>(100, 0.9, 0.1, 0.2, 1.0);
  model.addReservoir(res);

  SECTION("Throws when readout not set") {
    REQUIRE_THROWS(model.partialFit(Eigen::MatrixXd::Random(1, 1), Eigen::MatrixXd::Random(1, 1)));
  }

  // Use RLS for online updates
  auto readout = std::make_shared<RlsReadout>(0.99, 1.0, true);
  model.setReadout(readout);

  SECTION("Updates weights on partialFit") {
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 1);
    Eigen::MatrixXd target = Eigen::MatrixXd::Random(1, 1);

    // Initialize with a dummy fit to ensure W_out is allocated
    model.partialFit(input, target);

    // Get prediction after initial fit
    Eigen::MatrixXd pred_before = model.predict(input, true);

    // Perform another partial fit
    model.partialFit(input, target);

    // Prediction should have changed
    Eigen::MatrixXd pred_after = model.predict(input, true);
    REQUIRE(pred_before(0, 0) != Catch::Approx(pred_after(0, 0)));
  }

  SECTION("Parallel connection in partialFit") {
    Model pmodel;
    auto res1 = std::make_shared<RandomSparseReservoir>(50, 0.9, 0.1, 0.2, 1.0);
    auto res2 = std::make_shared<RandomSparseReservoir>(50, 0.9, 0.1, 0.2, 1.0);
    pmodel.addReservoir(res1, "parallel");
    pmodel.addReservoir(res2, "parallel");
    pmodel.setReadout(std::make_shared<RlsReadout>());

    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 1);
    Eigen::MatrixXd target = Eigen::MatrixXd::Random(1, 1);

    REQUIRE_NOTHROW(pmodel.partialFit(input, target));
  }
}

TEST_CASE("Model - reservoir count and connection type", "[Model]") {
  Model model;
  REQUIRE(model.getNumReservoirs() == 0);
  REQUIRE(model.getConnectionType() == "serial");

  model.addReservoir(std::make_shared<NvarReservoir>(2), "parallel");
  model.addReservoir(std::make_shared<NvarReservoir>(3), "parallel");
  REQUIRE(model.getNumReservoirs() == 2);
  REQUIRE(model.getConnectionType() == "parallel");
}

namespace {

Eigen::MatrixXd sine(int rows, double phase) {
  Eigen::MatrixXd values(rows, 1);
  for (int i = 0; i < rows; ++i) {
    values(i, 0) = std::sin(0.3 * i + phase);
  }
  return values;
}

// An independent copy with bit-identical parameters and states.
Model cloneModel(const Model &model) {
  std::stringstream buffer;
  model.save(buffer);
  return Model::load(buffer);
}

// An untrained model: 0 = RandomSparse, 1 = serial RandomSparse -> NVAR, 2 = parallel
// RandomSparse + NVAR. Every call builds identical reservoirs.
Model makeModel(int topology) {
  Model model;
  if (topology == 0) {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(30, 0.9, 0.3, 0.5, 1.0, true, 7));
  } else if (topology == 1) {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(8, 0.9, 0.5, 0.5, 1.0, true, 7));
    model.addReservoir(std::make_shared<NvarReservoir>(2, 2));
  } else {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(30, 0.9, 0.3, 0.5, 1.0, true, 7), "parallel");
    model.addReservoir(std::make_shared<NvarReservoir>(2, 2), "parallel");
  }
  model.setReadout(std::make_shared<RidgeReadout>(1e-4));
  return model;
}

// A model trained to predict the next value of a sine wave, so its outputs can be fed back.
Model makeGenerativeModel(int topology, int output_columns = 1) {
  Model model = makeModel(topology);
  const Eigen::MatrixXd series = sine(81, 0.0);
  model.fit(series.topRows(80), series.bottomRows(80).replicate(1, output_columns), 10);
  return model;
}

bool sameStates(const Model &a, const Model &b) {
  for (size_t i = 0; i < a.getNumReservoirs(); ++i) {
    if (a.getReservoir(i)->getState() != b.getReservoir(i)->getState()) {
      return false;
    }
  }
  return true;
}

} // namespace

TEST_CASE("Model - generative prediction in chunks equals one call", "[Model]") {
  const int topology = GENERATE(0, 1, 2); // serial, serial RandomSparse -> NVAR, parallel
  const auto chunks = GENERATE(std::vector<int>{3, 4}, std::vector<int>{6, 1}, std::vector<int>{1, 1, 1, 1, 1, 1, 1});
  const Model trained = makeGenerativeModel(topology);
  const Eigen::MatrixXd prime = sine(10, 2.0);

  Model whole = cloneModel(trained);
  const Eigen::MatrixXd expected = whole.predictGenerative(prime, 7);

  // The first chunk is primed; later chunks continue from the reservoir states.
  Model chunked = cloneModel(trained);
  Eigen::MatrixXd generated(0, 1);
  for (size_t i = 0; i < chunks.size(); ++i) {
    const Eigen::MatrixXd part = chunked.predictGenerative(i == 0 ? prime : Eigen::MatrixXd(0, 1), chunks[i]);
    generated.conservativeResize(generated.rows() + part.rows(), 1);
    generated.bottomRows(part.rows()) = part;
  }
  REQUIRE(generated == expected);
  REQUIRE(sameStates(chunked, whole));
}

TEST_CASE("Model - one-step generation advances the reservoirs", "[Model]") {
  Model model = makeGenerativeModel(0);
  model.predictGenerative(sine(10, 2.0), 1);
  const Eigen::MatrixXd state_before = model.getReservoir(0)->getState();
  const Eigen::MatrixXd first = model.predictGenerative(Eigen::MatrixXd(0, 1), 1);
  REQUIRE(model.getReservoir(0)->getState() != state_before);
  // The next one-step call does not repeat the previous output.
  REQUIRE(model.predictGenerative(Eigen::MatrixXd(0, 1), 1) != first);
}

TEST_CASE("Model - zero-step generation feeds nothing back", "[Model]") {
  const Model trained = makeGenerativeModel(2);
  const Eigen::MatrixXd prime = sine(10, 2.0);

  SECTION("Priming data is still consumed") {
    Model generative = cloneModel(trained);
    Model online = cloneModel(trained);
    REQUIRE(generative.predictGenerative(prime, 0).rows() == 0);
    online.predictOnline(prime); // advances through the same inputs without feedback
    REQUIRE(sameStates(generative, online));
  }
  SECTION("Without priming data the states are unchanged") {
    Model generative = cloneModel(trained);
    REQUIRE(generative.predictGenerative(Eigen::MatrixXd(0, 1), 0).rows() == 0);
    REQUIRE(sameStates(generative, trained));
  }
}

TEST_CASE("Model - generated outputs that cannot be fed back throw", "[Model]") {
  // Two output columns cannot be fed back into reservoirs locked to one input column.
  const int topology = GENERATE(0, 2);
  const int n_steps = GENERATE(1, 3);
  Model model = makeGenerativeModel(topology, 2);
  REQUIRE_THROWS_AS(model.predictGenerative(sine(10, 2.0), n_steps), std::invalid_argument);
  REQUIRE(model.predictGenerative(sine(10, 2.0), 0).cols() == 2);
}

namespace {

Eigen::MatrixXd readoutWeights(const Model &model) {
  return std::dynamic_pointer_cast<RidgeReadout>(model.getReadout())->getWeights();
}

std::vector<Eigen::MatrixXd> reservoirStates(const Model &model) {
  std::vector<Eigen::MatrixXd> states;
  for (size_t r = 0; r < model.getNumReservoirs(); ++r) {
    states.push_back(model.getReservoir(r)->getState());
  }
  return states;
}

// States for `inputs` from reset reservoirs, collected with the public Reservoir API and
// combined as the connection type says: an independent reference for Model.
Eigen::MatrixXd referenceStates(const Model &model, const Eigen::MatrixXd &inputs) {
  const bool serial = model.getConnectionType() == "serial";
  Eigen::MatrixXd chained = inputs;
  std::vector<Eigen::MatrixXd> outputs;
  for (size_t r = 0; r < model.getNumReservoirs(); ++r) {
    const auto reservoir = model.getReservoir(r);
    const Eigen::MatrixXd source = serial ? chained : inputs;
    Eigen::MatrixXd states(source.rows(), reservoir->getOutputDim(static_cast<int>(source.cols())));
    reservoir->resetState();
    for (Eigen::Index t = 0; t < source.rows(); ++t) {
      states.row(t) = reservoir->advance(source.row(t));
    }
    if (serial) {
      chained = states;
    } else {
      outputs.push_back(states);
    }
  }
  if (serial) {
    return chained;
  }
  Eigen::MatrixXd combined(inputs.rows(), 0);
  for (const auto &states : outputs) {
    combined.conservativeResize(Eigen::NoChange, combined.cols() + states.cols());
    combined.rightCols(states.cols()) = states;
  }
  return combined;
}

// Ridge weights fitted on the washed-out reference states of every sequence, stacked.
Eigen::MatrixXd referenceWeights(const Model &model, const std::vector<Eigen::MatrixXd> &inputs,
                                 const std::vector<Eigen::MatrixXd> &targets, int washout_len) {
  Eigen::MatrixXd states;
  Eigen::MatrixXd stacked_targets;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Eigen::MatrixXd sequence_states = referenceStates(model, inputs[i]);
    const Eigen::Index kept = sequence_states.rows() - washout_len;
    states.conservativeResize(states.rows() + kept, sequence_states.cols());
    states.bottomRows(kept) = sequence_states.bottomRows(kept);
    stacked_targets.conservativeResize(stacked_targets.rows() + kept, targets[i].cols());
    stacked_targets.bottomRows(kept) = targets[i].bottomRows(kept);
  }
  RidgeReadout reference(1e-4); // as in makeModel
  reference.fit(states, stacked_targets);
  return reference.getWeights();
}

std::vector<Eigen::MatrixXd> episodes(double phase_offset) {
  return {sine(20, phase_offset), sine(25, 1.0 + phase_offset), sine(15, 2.0 + phase_offset)};
}

} // namespace

TEST_CASE("Model - fitSequences with one sequence equals fit", "[Model]") {
  const int topology = GENERATE(0, 1, 2);
  const int washout_len = GENERATE(0, 5);
  CAPTURE(topology, washout_len);
  const Eigen::MatrixXd inputs = sine(40, 0.0);
  const Eigen::MatrixXd targets = sine(40, 0.3);

  Model single = makeModel(topology);
  Model sequences = makeModel(topology);
  single.fit(inputs, targets, washout_len);
  sequences.fitSequences({inputs}, {targets}, washout_len);

  REQUIRE(readoutWeights(sequences) == readoutWeights(single));
  REQUIRE(sameStates(sequences, single));
  const Eigen::MatrixXd probe = sine(10, 2.0);
  REQUIRE(sequences.predict(probe) == single.predict(probe));
}

TEST_CASE("Model - fitSequences fits one readout on per-sequence states", "[Model]") {
  const int topology = GENERATE(0, 1, 2);
  const int washout_len = GENERATE(0, 3);
  CAPTURE(topology, washout_len);
  const std::vector<Eigen::MatrixXd> inputs = episodes(0.0);
  const std::vector<Eigen::MatrixXd> targets = episodes(0.3);

  Model model = makeModel(topology);
  model.fitSequences(inputs, targets, washout_len);
  const Eigen::MatrixXd weights = readoutWeights(model);
  const std::vector<Eigen::MatrixXd> final_states = reservoirStates(model);

  // Each sequence starts from reset reservoirs and loses its own washout.
  REQUIRE(weights == referenceWeights(model, inputs, targets, washout_len));
  // The reference ran the last sequence last too, so both end in the same states.
  REQUIRE(reservoirStates(model) == final_states);
}

TEST_CASE("Model - fitSequences learns nothing across sequence boundaries", "[Model]") {
  const int topology = GENERATE(0, 1, 2);
  CAPTURE(topology);
  std::vector<Eigen::MatrixXd> inputs = episodes(0.0);
  std::vector<Eigen::MatrixXd> targets = episodes(0.3);

  Model forward = makeModel(topology);
  forward.fitSequences(inputs, targets, 3);
  std::reverse(inputs.begin(), inputs.end());
  std::reverse(targets.begin(), targets.end());
  Model backward = makeModel(topology);
  backward.fitSequences(inputs, targets, 3);

  // The order of the sequences only changes the summation order of the fit.
  const Eigen::MatrixXd probe = sine(30, 0.7);
  REQUIRE(backward.predict(probe).isApprox(forward.predict(probe), 1e-9));
}

TEST_CASE("Model - fitSequences rejects invalid sequences and changes nothing", "[Model]") {
  Model model = makeModel(1);
  model.fit(sine(20, 0.0), sine(20, 0.3));
  const Eigen::MatrixXd weights_before = readoutWeights(model);
  const std::vector<Eigen::MatrixXd> states_before = reservoirStates(model);

  std::vector<Eigen::MatrixXd> inputs = {sine(10, 0.0), sine(12, 1.0)};
  std::vector<Eigen::MatrixXd> targets = {sine(10, 0.3), sine(12, 1.3)};
  int washout_len = 2;
  bool out_of_range = false;
  std::string message;

  SECTION("No sequences") {
    inputs.clear();
    targets.clear();
    message = "inputs must hold at least one sequence";
  }
  SECTION("Different numbers of input and target sequences") {
    targets.pop_back();
    message = "targets must hold as many sequences as inputs";
  }
  SECTION("Empty inputs") {
    inputs[1] = Eigen::MatrixXd(0, 1);
    targets[1] = Eigen::MatrixXd(0, 1);
    message = "sequence 1: inputs must be a non-empty 2D matrix";
  }
  SECTION("Inputs without columns") {
    inputs[1] = Eigen::MatrixXd(12, 0);
    message = "sequence 1: inputs must be a non-empty 2D matrix";
  }
  SECTION("Different row counts") {
    targets[1] = sine(11, 1.3);
    message = "sequence 1: targets must have the same number of rows as inputs";
  }
  SECTION("Targets without columns") {
    targets[1] = Eigen::MatrixXd(12, 0);
    message = "sequence 1: targets must have at least one column";
  }
  SECTION("Negative washout") {
    washout_len = -1;
    out_of_range = true;
    message = "washout_len must be non-negative";
  }
  SECTION("A sequence no longer than the washout") {
    inputs[1] = sine(2, 1.0);
    targets[1] = sine(2, 1.3);
    out_of_range = true;
    message = "sequence 1: washout_len must be non-negative and less than the number of input rows";
  }
  SECTION("Different input widths") {
    inputs[1] = Eigen::MatrixXd::Ones(12, 2);
    message = "sequence 1: inputs must have as many columns as sequence 0";
  }
  SECTION("Different target widths") {
    targets[1] = Eigen::MatrixXd::Ones(12, 2);
    message = "sequence 1: targets must have as many columns as sequence 0";
  }

  CAPTURE(message);
  if (out_of_range) {
    REQUIRE_THROWS_AS(model.fitSequences(inputs, targets, washout_len), std::out_of_range);
  } else {
    REQUIRE_THROWS_AS(model.fitSequences(inputs, targets, washout_len), std::invalid_argument);
  }
  REQUIRE_THROWS_WITH(model.fitSequences(inputs, targets, washout_len), Catch::Matchers::StartsWith(message));
  REQUIRE(readoutWeights(model) == weights_before);
  REQUIRE(reservoirStates(model) == states_before);
}

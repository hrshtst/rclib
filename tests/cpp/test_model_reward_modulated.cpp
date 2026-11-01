#include "rclib/Model.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RewardModulatedReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"
#include "rclib/reservoirs/RewardModulatedReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using Rule = RewardModulatedHebbianRule;

Eigen::MatrixXd ramp(Eigen::Index rows, Eigen::Index cols, double phase) {
  Eigen::MatrixXd values(rows, cols);
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    values(i) = std::sin(0.3 * static_cast<double>(i) + phase);
  }
  return values;
}

Rule onlineRule() {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.baseline = Rule::NONE;
  rule.learning_rate = 0.1;
  rule.noise_amplitude = 0.2;
  return rule;
}

struct RewardModel {
  Model model;
  std::shared_ptr<RandomSparseReservoir> reservoir;
  std::shared_ptr<RewardModulatedReadout> readout;
};

RewardModel makeRewardModel(const std::string &connection = "serial") {
  RewardModel built;
  built.reservoir = std::make_shared<RandomSparseReservoir>(20, 0.9, 0.3, 0.5, 1.0, true, 3);
  built.model.addReservoir(built.reservoir, connection);
  built.readout = std::make_shared<RewardModulatedReadout>(2, onlineRule(), true);
  built.model.setReadout(built.readout);
  return built;
}

} // namespace

TEST_CASE("Model - act explores through the readout and reinforce updates it", "[Model][reward]") {
  RewardModel built = makeRewardModel();
  const Eigen::MatrixXd outputs = built.model.act(ramp(3, 1, 0.0));
  REQUIRE(outputs.rows() == 3);
  REQUIRE(outputs.cols() == 2);
  // Zero weights at the start, so the outputs are the exploration noise.
  REQUIRE(outputs.cwiseAbs().maxCoeff() <= 0.2);
  REQUIRE(built.readout->getLearning().hasEligibility());

  const Eigen::MatrixXd eligibility = built.readout->getLearning().getEligibility();
  built.model.reinforce(2.0);
  REQUIRE(built.readout->getWeights().isApprox(0.1 * 2.0 * eligibility, 1e-15));
  REQUIRE_THROWS_AS(built.model.reinforce(1.0), std::logic_error);
  built.model.act(ramp(1, 1, 0.5));
  built.model.modulate(-1.0);
  REQUIRE(built.readout->getLearning().getUpdateCount() == 2);
  built.model.act(ramp(1, 1, 0.7));
  built.model.clearEligibility();
  REQUIRE_FALSE(built.readout->getLearning().hasEligibility());
}

TEST_CASE("Model - act without a readout returns the reservoir states", "[Model][reward]") {
  Model model;
  model.addReservoir(std::make_shared<RandomSparseReservoir>(6, 0.9, 0.5, 1.0, 1.0, false, 4));
  model.addReservoir(std::make_shared<RandomSparseReservoir>(5, 0.9, 0.5, 1.0, 1.0, false, 5));
  REQUIRE_FALSE(model.hasReadout());
  const Eigen::MatrixXd states = model.act(ramp(4, 2, 0.0));
  REQUIRE(states.rows() == 4);
  REQUIRE(states.cols() == 5);
  REQUIRE(Eigen::MatrixXd(states.row(3)) == model.getReservoir(1)->getState());

  SECTION("a model without a reward-modulated component cannot reinforce") {
    REQUIRE_THROWS_AS(model.reinforce(1.0), std::logic_error);
    REQUIRE_THROWS_AS(model.modulate(1.0), std::logic_error);
    REQUIRE_THROWS_AS(model.clearEligibility(), std::logic_error);
  }
  SECTION("but rejects a non-finite reward first") {
    REQUIRE_THROWS_AS(model.reinforce(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
  }
}

TEST_CASE("Model - noise-free act equals predictOnline", "[Model][reward]") {
  const std::string connection = GENERATE(std::string("serial"), std::string("parallel"));
  const auto build = [&](Model &model) {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(15, 0.9, 0.3, 0.4, 1.0, true, 6), connection);
    model.addReservoir(std::make_shared<NvarReservoir>(2, 2), connection);
  };
  const Eigen::MatrixXd inputs = ramp(6, 2, 0.3);

  SECTION("with a reward-modulated readout, bit for bit") {
    Model acting;
    Model predicting;
    build(acting);
    build(predicting);
    NvarReservoir nvar(2, 2);
    const int features = connection == "serial" ? nvar.getOutputDim(15) : 15 + nvar.getOutputDim(2);
    Eigen::MatrixXd weights = ramp(features + 1, 2, 0.1);
    for (Model *model : {&acting, &predicting}) {
      auto readout = std::make_shared<RewardModulatedReadout>(2, onlineRule(), true);
      readout->setWeights(weights);
      model->setReadout(readout);
    }
    REQUIRE(acting.act(inputs, false) == predicting.predictOnline(inputs));
  }
  SECTION("with a supervised readout") {
    Model acting;
    Model predicting;
    for (Model *model : {&acting, &predicting}) {
      build(*model);
      model->setReadout(std::make_shared<RidgeReadout>(1e-3, true));
      model->fit(ramp(30, 2, 1.0), ramp(30, 1, 2.0));
      model->resetReservoirs();
    }
    REQUIRE(acting.act(inputs, false).isApprox(predicting.predictOnline(inputs), 1e-12));
  }
}

TEST_CASE("Model - act checks everything before the model changes", "[Model][reward]") {
  RewardModel built = makeRewardModel();
  built.model.act(ramp(2, 1, 0.0));
  const Eigen::MatrixXd state = built.reservoir->getState();
  const Eigen::MatrixXd eligibility = built.readout->getLearning().getEligibility();

  SECTION("non-finite inputs") {
    Eigen::MatrixXd inputs = ramp(3, 1, 0.0);
    inputs(2, 0) = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(built.model.act(inputs), std::invalid_argument);
  }
  SECTION("inputs of another width") { REQUIRE_THROWS_AS(built.model.act(ramp(1, 2, 0.0)), std::invalid_argument); }
  SECTION("a readout locked to another feature width") {
    auto readout = std::make_shared<RewardModulatedReadout>(2, onlineRule(), true);
    readout->setWeights(Eigen::MatrixXd::Zero(8, 2));
    built.model.setReadout(readout);
    REQUIRE_THROWS_AS(built.model.act(ramp(1, 1, 0.0)), std::invalid_argument);
  }
  SECTION("supervised training of a reward-modulated readout") {
    REQUIRE_THROWS_AS(built.model.fit(ramp(5, 1, 0.0), ramp(5, 2, 0.0)), std::logic_error);
    REQUIRE_THROWS_AS(built.model.fitSequences({ramp(5, 1, 0.0)}, {ramp(5, 2, 0.0)}), std::logic_error);
    REQUIRE_THROWS_AS(built.model.partialFit(ramp(1, 1, 0.0), ramp(1, 2, 0.0)), std::logic_error);
  }
  REQUIRE(built.reservoir->getState() == state);
  REQUIRE(built.readout->getLearning().getEligibility() == eligibility);
}

TEST_CASE("Model - act rejects a reservoir listed twice", "[Model][reward]") {
  // Width 5 in and 5 out, so a serial model can feed the reservoir its own output.
  auto shared = std::make_shared<RandomSparseReservoir>(5, 0.9, 0.5, 0.5, 1.0, true, 7);
  const std::string connection = GENERATE(std::string("serial"), std::string("parallel"));
  Model model;
  model.addReservoir(shared, connection);
  model.addReservoir(shared, connection);
  model.setReadout(std::make_shared<RewardModulatedReadout>(1, onlineRule(), true));
  REQUIRE_THROWS_AS(model.act(ramp(1, 5, 0.0)), std::logic_error);
  REQUIRE(shared->getInputDim() == 0); // never advanced
  if (connection == "serial") {
    // The existing methods keep stepping the object once per occurrence.
    Model supervised;
    supervised.addReservoir(shared);
    supervised.addReservoir(shared);
    supervised.setReadout(std::make_shared<LmsReadout>(0.01, true));
    supervised.partialFit(ramp(1, 5, 0.0), ramp(1, 1, 0.0));
    REQUIRE_NOTHROW(supervised.predictOnline(ramp(2, 5, 0.1)));
  }
}

TEST_CASE("Model - a failing row changes no learning state and keeps the rows before it", "[Model][reward]") {
  // An identity NVAR passes the inputs to the readout, so the second row's output overflows.
  Model failing;
  Model reference;
  std::shared_ptr<RewardModulatedReadout> readouts[2];
  int i = 0;
  for (Model *model : {&failing, &reference}) {
    model->addReservoir(std::make_shared<NvarReservoir>(1, 1));
    readouts[i] = std::make_shared<RewardModulatedReadout>(1, onlineRule(), true);
    readouts[i]->setWeights(Eigen::MatrixXd::Constant(2, 1, 10.0));
    model->setReadout(readouts[i++]);
  }
  Eigen::MatrixXd inputs(2, 1);
  inputs << 1.0, 1e308;
  REQUIRE_THROWS_AS(failing.act(inputs), std::runtime_error);
  reference.act(inputs.topRows(1));
  REQUIRE(readouts[0]->getLearning().hasEligibility());
  REQUIRE(readouts[0]->getLearning().getEligibility() == readouts[1]->getLearning().getEligibility());
  REQUIRE(readouts[0]->getWeights() == readouts[1]->getWeights());
}

TEST_CASE("Model - resetReservoirs restarts the readout's post-synaptic average", "[Model][reward]") {
  Model model;
  model.addReservoir(std::make_shared<RandomSparseReservoir>(10, 0.9, 0.5, 0.5, 1.0, true, 8));
  Rule rule = onlineRule();
  rule.post = Rule::HIGH_PASS;
  auto readout = std::make_shared<RewardModulatedReadout>(1, rule, true);
  model.setReadout(readout);
  model.act(ramp(3, 1, 0.0));
  REQUIRE(readout->getLearning().isPostAverageInitialized());
  model.resetReservoirs();
  REQUIRE_FALSE(readout->getLearning().isPostAverageInitialized());
  model.act(ramp(1, 1, 0.4));
  REQUIRE(readout->getLearning().getEligibility().isZero(0.0));
}

namespace {

std::shared_ptr<RewardModulatedReservoir> plasticReservoir(int n, unsigned int seed, const Rule &rule = onlineRule()) {
  return std::make_shared<RewardModulatedReservoir>(n, 0.9, 0.5, 0.5, 1.0, true, seed,
                                                    RandomSparseReservoir::POWER_ITERATION, 1.0, rule);
}

} // namespace

TEST_CASE("Model - a reward-modulated reservoir listed twice is rejected", "[Model][reward]") {
  auto shared = plasticReservoir(5, 11);
  const std::string connection = GENERATE(std::string("serial"), std::string("parallel"));
  Model model;
  model.addReservoir(shared, connection);
  model.addReservoir(shared, connection);
  const Eigen::MatrixXd weights = shared->getRecurrentWeights();
  REQUIRE_THROWS_AS(model.act(ramp(1, 5, 0.0)), std::logic_error);
  REQUIRE_THROWS_AS(model.reinforce(1.0), std::logic_error);
  REQUIRE_THROWS_AS(model.modulate(1.0), std::logic_error);
  REQUIRE_THROWS_AS(model.clearEligibility(), std::logic_error);
  REQUIRE(Eigen::MatrixXd(shared->getRecurrentWeights()) == weights);
  REQUIRE_FALSE(shared->getLearning().hasEligibility());
  REQUIRE(shared->getInputDim() == 0);
}

TEST_CASE("Model - an update fails on all learners when one fails", "[Model][reward]") {
  auto reservoir = plasticReservoir(8, 12);
  Rule huge = onlineRule();
  huge.learning_rate = 1e308;
  auto readout = std::make_shared<RewardModulatedReadout>(1, huge, true);
  Model model;
  model.addReservoir(reservoir);
  model.setReadout(readout);
  model.act(ramp(3, 1, 0.0));
  const Eigen::MatrixXd recurrent = reservoir->getRecurrentWeights();
  const Eigen::MatrixXd eligibility = reservoir->getLearning().getEligibility();
  const Eigen::MatrixXd readout_weights = readout->getWeights();
  // The reservoir's update is fine, the readout's overflows, so neither applies.
  REQUIRE_THROWS_AS(model.reinforce(5.0), std::runtime_error);
  REQUIRE(Eigen::MatrixXd(reservoir->getRecurrentWeights()) == recurrent);
  REQUIRE(reservoir->getLearning().getEligibility() == eligibility);
  REQUIRE(reservoir->getLearning().hasEligibility());
  REQUIRE(reservoir->getLearning().getUpdateCount() == 0);
  REQUIRE(readout->getWeights() == readout_weights);
  REQUIRE(readout->getLearning().getUpdateCount() == 0);
}

TEST_CASE("Model - parallel reward-modulated reservoirs act and learn as they do alone", "[Model][reward]") {
  Model model;
  auto first = plasticReservoir(12, 13);
  auto second = plasticReservoir(9, 14);
  model.addReservoir(first, "parallel");
  model.addReservoir(second, "parallel");
  auto first_alone = plasticReservoir(12, 13);
  auto second_alone = plasticReservoir(9, 14);

  const Eigen::MatrixXd inputs = ramp(4, 2, 0.2);
  const Eigen::MatrixXd states = model.act(inputs);
  REQUIRE(states.leftCols(12) == first_alone->act(inputs));
  REQUIRE(states.rightCols(9) == second_alone->act(inputs));
  model.reinforce(0.5);
  first_alone->reinforce(0.5);
  second_alone->reinforce(0.5);
  REQUIRE(Eigen::MatrixXd(first->getRecurrentWeights()) == Eigen::MatrixXd(first_alone->getRecurrentWeights()));
  REQUIRE(Eigen::MatrixXd(second->getRecurrentWeights()) == Eigen::MatrixXd(second_alone->getRecurrentWeights()));
  REQUIRE(model.act(inputs, false) ==
          (Eigen::MatrixXd(4, 21) << first_alone->act(inputs, false), second_alone->act(inputs, false)).finished());
}

TEST_CASE("Model - a model without a readout learns in its reservoir", "[Model][reward]") {
  Model model;
  auto reservoir = plasticReservoir(10, 15);
  model.addReservoir(reservoir);
  const Eigen::MatrixXd recurrent = reservoir->getRecurrentWeights();
  const Eigen::MatrixXd rates = model.act(ramp(3, 1, 0.0));
  REQUIRE(rates.cols() == 10);
  model.reinforce(1.0);
  REQUIRE(Eigen::MatrixXd(reservoir->getRecurrentWeights()) != recurrent);
  model.act(ramp(2, 1, 0.3));
  model.resetReservoirs();
  REQUIRE(reservoir->getState().isZero(0.0));
  REQUIRE(reservoir->getLearning().hasEligibility()); // a reset keeps the eligibility
}

namespace {

// A conventional readout whose finite-looking predictions overflow.
class OverflowingReadout : public Readout {
public:
  void fit(const Eigen::MatrixXd & /*states*/, const Eigen::MatrixXd & /*targets*/) override {}
  void partialFit(const Eigen::MatrixXd & /*state*/, const Eigen::MatrixXd & /*target*/) override {}
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override {
    return Eigen::MatrixXd::Constant(states.rows(), 1, -std::numeric_limits<double>::infinity());
  }
};

} // namespace

TEST_CASE("Model - a non-finite output from any readout fails the row", "[Model][reward]") {
  const bool explore = GENERATE(true, false);
  SECTION("a conventional readout after a reward-modulated reservoir") {
    auto reservoir = plasticReservoir(8, 16);
    Model model;
    model.addReservoir(reservoir);
    model.setReadout(std::make_shared<OverflowingReadout>());
    REQUIRE_THROWS_AS(model.act(ramp(2, 1, 0.0), explore), std::runtime_error);
    // The reservoir staged the row, but nothing was committed.
    REQUIRE_FALSE(reservoir->getLearning().hasEligibility());
    REQUIRE(reservoir->getLearning().getEligibility().isZero(0.0));
    REQUIRE(reservoir->getLearning().getEligibilityBound() == 0.0);
  }
  SECTION("reservoir states without a readout") {
    Model model;
    model.addReservoir(std::make_shared<NvarReservoir>(1, 2)); // squares 1e200
    REQUIRE_THROWS_AS(model.act(Eigen::MatrixXd::Constant(1, 1, 1e200), explore), std::runtime_error);
  }
}

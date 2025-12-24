#include "rclib/Model.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <memory>

TEST_CASE("Model - configuration", "[Model]") {
  Model model;

  SECTION("Throws when not configured") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_THROWS(model.fit(inputs, targets));
    REQUIRE_THROWS(model.predict(inputs));
    REQUIRE_THROWS(model.predictOnline(inputs.row(0)));
  }

  auto res = std::make_shared<RandomSparseReservoir>(10, 0.9, 0.5, 0.1, 1.0);
  model.addReservoir(res);

  SECTION("Throws when readout is not set") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_THROWS(model.fit(inputs, targets));
    REQUIRE_THROWS(model.predict(inputs));
    REQUIRE_THROWS(model.predictOnline(inputs.row(0)));
  }

  auto readout = std::make_shared<RidgeReadout>();
  model.setReadout(readout);

  SECTION("Does not throw when fully configured") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_NOTHROW(model.fit(inputs, targets));
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

#include "rclib/reservoirs/NvarReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <stdexcept>

TEST_CASE("NvarReservoir - Constructor and Initialization", "[NvarReservoir]") {
  int num_lags = 3;
  NvarReservoir res(num_lags);

  SECTION("State is initialized to zeros before first advance") { REQUIRE(res.getState().isZero(0)); }
}

TEST_CASE("NvarReservoir - State Advancement", "[NvarReservoir]") {
  int num_lags = 3;
  int input_dim = 2;
  NvarReservoir res(num_lags);

  Eigen::MatrixXd input1 = Eigen::MatrixXd::Random(1, input_dim);
  Eigen::MatrixXd input2 = Eigen::MatrixXd::Random(1, input_dim);
  Eigen::MatrixXd input3 = Eigen::MatrixXd::Random(1, input_dim);

  res.advance(input1);
  REQUIRE(res.getState().block(0, 0, 1, input_dim) == input1);
  REQUIRE(res.getState().block(0, input_dim, 1, input_dim).isZero(0));
  REQUIRE(res.getState().block(0, 2 * input_dim, 1, input_dim).isZero(0));

  res.advance(input2);
  REQUIRE(res.getState().block(0, 0, 1, input_dim) == input2);
  REQUIRE(res.getState().block(0, input_dim, 1, input_dim) == input1);
  REQUIRE(res.getState().block(0, 2 * input_dim, 1, input_dim).isZero(0));

  res.advance(input3);
  REQUIRE(res.getState().block(0, 0, 1, input_dim) == input3);
  REQUIRE(res.getState().block(0, input_dim, 1, input_dim) == input2);
  REQUIRE(res.getState().block(0, 2 * input_dim, 1, input_dim) == input1);
}

TEST_CASE("NvarReservoir - Polynomial Features", "[NvarReservoir]") {
  NvarReservoir res(2, 2);

  Eigen::MatrixXd input1(1, 2);
  input1 << 1.0, 2.0;
  res.advance(input1);

  Eigen::MatrixXd expected1(1, 14);
  expected1 << 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  REQUIRE(res.getState().isApprox(expected1));

  Eigen::MatrixXd input2(1, 2);
  input2 << 3.0, 4.0;
  res.advance(input2);

  Eigen::MatrixXd expected2(1, 14);
  expected2 << 3.0, 4.0, 1.0, 2.0, 9.0, 12.0, 3.0, 6.0, 16.0, 4.0, 8.0, 1.0, 2.0, 4.0;
  REQUIRE(res.getState().isApprox(expected2));
  REQUIRE(res.getOutputDim(2) == 14);
}

TEST_CASE("NvarReservoir - Validation", "[NvarReservoir]") {
  REQUIRE_THROWS_AS(NvarReservoir(0), std::invalid_argument);
  REQUIRE_THROWS_AS(NvarReservoir(1, 0), std::invalid_argument);

  NvarReservoir res(1);
  Eigen::MatrixXd batch = Eigen::MatrixXd::Zero(2, 1);
  REQUIRE_THROWS_AS(res.advance(batch), std::invalid_argument);

  NvarReservoir large_res(10, 8);
  REQUIRE_THROWS_AS(large_res.getOutputDim(10), std::length_error);
}

TEST_CASE("NvarReservoir - State Reset", "[NvarReservoir]") {
  int num_lags = 3;
  int input_dim = 2;
  NvarReservoir res(num_lags);

  Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, input_dim);

  res.advance(input);
  REQUIRE_FALSE(res.getState().isZero(0));

  res.resetState();
  REQUIRE(res.getState().isZero(0));
}

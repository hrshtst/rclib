#include "rclib/reservoirs/NvarReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

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

#pragma once

#include "Readout.h"
#include "Reservoir.h"

#include <memory>
#include <string>
#include <vector>

class Model {
public:
  void addReservoir(std::shared_ptr<Reservoir> res, std::string connection_type = "serial");
  void setReadout(std::shared_ptr<Readout> readout);
  void fit(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, int washout_len = 0);
  Eigen::MatrixXd predict(const Eigen::MatrixXd &inputs, bool reset_state_before_predict = true);
  Eigen::MatrixXd predictOnline(const Eigen::MatrixXd &input);
  Eigen::MatrixXd predictGenerative(const Eigen::MatrixXd &prime_inputs, int n_steps);
  void resetReservoirs();

  std::shared_ptr<Reservoir> getReservoir(size_t index) const;
  std::shared_ptr<Readout> getReadout() const;

private:
  std::vector<std::shared_ptr<Reservoir>> reservoirs;
  std::shared_ptr<Readout> readout;
  std::string connection_type = "serial";
};

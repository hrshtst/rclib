#pragma once

#include "Reservoir.h"
#include "Readout.h"
#include <vector>
#include <memory>
#include <string>

class Model {
public:
    void addReservoir(std::shared_ptr<Reservoir> res, std::string connection_type = "serial");
    void setReadout(std::shared_ptr<Readout> readout);
    void fit(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, int washout_len = 0);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& inputs);
    Eigen::MatrixXd predictOnline(const Eigen::MatrixXd& input);

private:
    std::vector<std::shared_ptr<Reservoir>> reservoirs;
    std::shared_ptr<Readout> readout;
    std::string connection_type = "serial";
};

#include "rcl/Model.h"
#include <stdexcept>

void Model::addReservoir(std::shared_ptr<Reservoir> res, std::string connection_type) {
    reservoirs.push_back(res);
    this->connection_type = connection_type;
}

void Model::setReadout(std::shared_ptr<Readout> readout) {
    this->readout = readout;
}

void Model::fit(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, int washout_len) {
    if (reservoirs.empty() || !readout) {
        throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
    }

    if (washout_len < 0 || washout_len >= inputs.rows()) {
        throw std::out_of_range("washout_len must be non-negative and less than the number of input rows.");
    }

    for (auto& res : reservoirs) {
        res->resetState();
    }

    // Collect states from all reservoirs for the entire input sequence
    Eigen::MatrixXd all_states_full;

    if (connection_type == "serial") {
        Eigen::MatrixXd current_input = inputs;
        for (const auto& res : reservoirs) {
            Eigen::MatrixXd res_states(inputs.rows(), res->getState().cols());
            for (int i = 0; i < inputs.rows(); ++i) {
                res_states.row(i) = res->advance(current_input.row(i));
            }
            current_input = res_states;
        }
        all_states_full = current_input;
    } else if (connection_type == "parallel") {
        for (const auto& res : reservoirs) {
            Eigen::MatrixXd res_states(inputs.rows(), res->getState().cols());
            for (int i = 0; i < inputs.rows(); ++i) {
                res_states.row(i) = res->advance(inputs.row(i));
            }
            if (all_states_full.size() == 0) {
                all_states_full = res_states;
            } else {
                all_states_full.conservativeResize(all_states_full.rows(), all_states_full.cols() + res_states.cols());
                all_states_full.rightCols(res_states.cols()) = res_states;
            }
        }
    }

    // Apply washout period
    Eigen::MatrixXd fit_states = all_states_full.bottomRows(all_states_full.rows() - washout_len);
    Eigen::MatrixXd fit_targets = targets.bottomRows(targets.rows() - washout_len);

    readout->fit(fit_states, fit_targets);
}

Eigen::MatrixXd Model::predict(const Eigen::MatrixXd& inputs) {
    if (reservoirs.empty() || !readout) {
        throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
    }

    for (auto& res : reservoirs) {
        res->resetState();
    }

    // Collect states from all reservoirs
    Eigen::MatrixXd all_states;

    if (connection_type == "serial") {
        Eigen::MatrixXd current_input = inputs;
        for (const auto& res : reservoirs) {
            Eigen::MatrixXd res_states(inputs.rows(), res->getState().cols());
            for (int i = 0; i < inputs.rows(); ++i) {
                res_states.row(i) = res->advance(current_input.row(i));
            }
            current_input = res_states;
        }
        all_states = current_input;
    } else if (connection_type == "parallel") {
        for (const auto& res : reservoirs) {
            Eigen::MatrixXd res_states(inputs.rows(), res->getState().cols());
            for (int i = 0; i < inputs.rows(); ++i) {
                res_states.row(i) = res->advance(inputs.row(i));
            }
            if (all_states.size() == 0) {
                all_states = res_states;
            } else {
                all_states.conservativeResize(all_states.rows(), all_states.cols() + res_states.cols());
                all_states.rightCols(res_states.cols()) = res_states;
            }
        }
    }

    return readout->predict(all_states);
}

Eigen::MatrixXd Model::predictOnline(const Eigen::MatrixXd& input) {
    if (reservoirs.empty() || !readout) {
        throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
    }

    // Collect states from all reservoirs
    Eigen::MatrixXd all_states;

    if (connection_type == "serial") {
        Eigen::MatrixXd current_input = input;
        for (const auto& res : reservoirs) {
            current_input = res->advance(current_input);
        }
        all_states = current_input;
    } else if (connection_type == "parallel") {
        for (const auto& res : reservoirs) {
            Eigen::MatrixXd res_state = res->advance(input);
            if (all_states.size() == 0) {
                all_states = res_state;
            } else {
                all_states.conservativeResize(all_states.rows(), all_states.cols() + res_state.cols());
                all_states.rightCols(res_state.cols()) = res_state;
            }
        }
    }

    return readout->predict(all_states);
}

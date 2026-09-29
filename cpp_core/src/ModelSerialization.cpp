#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>

namespace {

// Wire codes and type tags are part of the file format.
constexpr std::uint8_t serial_code = 0;
constexpr std::uint8_t parallel_code = 1;
constexpr const char *random_sparse_tag = "RandomSparseReservoir";
constexpr const char *nvar_tag = "NvarReservoir";
constexpr const char *ridge_tag = "RidgeReadout";
constexpr const char *rls_tag = "RlsReadout";
constexpr const char *lms_tag = "LmsReadout";

// Calls action(concrete, tag) with the reservoir cast to its built-in type. Types
// are matched exactly, so a user subclass of a built-in type is rejected rather
// than saved as its base.
template <typename Action> void visitReservoir(const Reservoir &reservoir, Action &&action) {
  if (typeid(reservoir) == typeid(RandomSparseReservoir)) {
    action(static_cast<const RandomSparseReservoir &>(reservoir), random_sparse_tag);
  } else if (typeid(reservoir) == typeid(NvarReservoir)) {
    action(static_cast<const NvarReservoir &>(reservoir), nvar_tag);
  } else {
    throw SerializationError("Model: unsupported reservoir type; only RandomSparseReservoir and NvarReservoir can be "
                             "saved.");
  }
}

template <typename Action> void visitReadout(const Readout &readout, Action &&action) {
  if (typeid(readout) == typeid(RidgeReadout)) {
    action(static_cast<const RidgeReadout &>(readout), ridge_tag);
  } else if (typeid(readout) == typeid(RlsReadout)) {
    action(static_cast<const RlsReadout &>(readout), rls_tag);
  } else if (typeid(readout) == typeid(LmsReadout)) {
    action(static_cast<const LmsReadout &>(readout), lms_tag);
  } else {
    throw SerializationError("Model: unsupported readout type; only RidgeReadout, RlsReadout and LmsReadout can be "
                             "saved.");
  }
}

std::shared_ptr<Reservoir> loadReservoir(BinaryReader &reader) {
  const std::string tag = reader.readString();
  if (tag == random_sparse_tag) {
    return RandomSparseReservoir::load(reader);
  }
  if (tag == nvar_tag) {
    return NvarReservoir::load(reader);
  }
  throw SerializationError("Model: unknown reservoir type tag '" + tag + "'.");
}

std::shared_ptr<Readout> loadReadout(BinaryReader &reader) {
  const std::string tag = reader.readString();
  if (tag == ridge_tag) {
    return RidgeReadout::load(reader);
  }
  if (tag == rls_tag) {
    return RlsReadout::load(reader);
  }
  if (tag == lms_tag) {
    return LmsReadout::load(reader);
  }
  throw SerializationError("Model: unknown readout type tag '" + tag + "'.");
}

// Threads widths through the model. getInputDim() is the width a component is
// locked to (0 = not fixed yet); an unknown width skips the checks that need it.
void checkTopology(const std::vector<std::shared_ptr<Reservoir>> &reservoirs, bool parallel, const Readout &readout) {
  std::int64_t features = 0; // width reaching the readout
  if (parallel) {
    int input = 0; // shared by every reservoir
    for (const auto &reservoir : reservoirs) {
      const int locked = reservoir->getInputDim();
      if (locked > 0 && input > 0 && locked != input) {
        throw SerializationError("Model: parallel reservoirs are locked to different input widths.");
      }
      if (locked > 0) {
        input = locked;
      }
    }
    if (input > 0) {
      std::vector<std::int64_t> widths;
      for (const auto &reservoir : reservoirs) {
        widths.push_back(reservoir->getOutputDim(input));
      }
      features = sumFeatureWidths(widths);
    }
  } else {
    int width = 0; // entering the current reservoir, then leaving it
    for (const auto &reservoir : reservoirs) {
      const int locked = reservoir->getInputDim();
      if (locked > 0 && width > 0 && locked != width) {
        throw SerializationError("Model: a serial reservoir is locked to an input width of " + std::to_string(locked) +
                                 " but the previous reservoir outputs " + std::to_string(width) + ".");
      }
      const int input = locked > 0 ? locked : width;
      width = input > 0 ? reservoir->getOutputDim(input) : 0;
    }
    features = sumFeatureWidths({width});
  }

  const int readout_width = readout.getInputDim();
  if (readout_width > 0 && features > 0 && readout_width != features) {
    throw SerializationError("Model: the readout is fitted to " + std::to_string(readout_width) +
                             " features but the reservoirs output " + std::to_string(features) + ".");
  }
}

// Checks shared by save and load, so that everything save() accepts load() accepts.
void checkModel(const std::vector<std::shared_ptr<Reservoir>> &reservoirs, const std::string &connection_type,
                const std::shared_ptr<Readout> &readout) {
  if (reservoirs.empty() || !readout) {
    throw SerializationError("Model: a model needs at least one reservoir and a readout.");
  }
  for (const auto &reservoir : reservoirs) {
    visitReservoir(*reservoir, [](const auto &concrete, const char * /*tag*/) { concrete.checkConsistency(); });
  }
  visitReadout(*readout, [](const auto &concrete, const char * /*tag*/) { concrete.checkConsistency(); });
  checkTopology(reservoirs, connection_type == "parallel", *readout);
}

} // namespace

void Model::save(std::ostream &os) const {
  translateSerializationErrors("Model", [&] {
    checkModel(reservoirs, connection_type, readout);
    // The format stores each reservoir by value; a shared one would load as two
    // independent objects and change how the model evolves.
    std::set<const Reservoir *> seen;
    for (const auto &reservoir : reservoirs) {
      if (!seen.insert(reservoir.get()).second) {
        throw SerializationError("Model: cannot save a model that holds the same reservoir object more than once.");
      }
    }

    BinaryWriter writer(os);
    writer.writeHeader();
    writer.writeU8(connection_type == "parallel" ? parallel_code : serial_code);
    writer.writeUInt(static_cast<std::uint32_t>(reservoirs.size()));
    for (const auto &reservoir : reservoirs) {
      visitReservoir(*reservoir, [&](const auto &concrete, const char *tag) {
        writer.writeString(tag);
        concrete.save(writer);
      });
    }
    visitReadout(*readout, [&](const auto &concrete, const char *tag) {
      writer.writeString(tag);
      concrete.save(writer);
    });
  });
}

Model Model::load(std::istream &is) {
  return translateSerializationErrors("Model", [&] {
    BinaryReader reader(is);
    reader.readHeader();

    Model model;
    const std::uint8_t connection_code = reader.readU8();
    if (connection_code != serial_code && connection_code != parallel_code) {
      throw SerializationError("Model: unknown connection code " + std::to_string(connection_code) + ".");
    }
    model.connection_type = connection_code == parallel_code ? "parallel" : "serial";
    // Each iteration consumes input, so a bogus count fails at the end of the data
    // instead of reserving memory up front.
    const std::uint32_t n_reservoirs = reader.readUInt();
    for (std::uint32_t i = 0; i < n_reservoirs; ++i) {
      model.reservoirs.push_back(loadReservoir(reader));
    }
    model.readout = loadReadout(reader);
    if (reader.remainingBytes() != 0) {
      throw SerializationError("Model: unexpected data after the end of the model.");
    }

    checkModel(model.reservoirs, model.connection_type, model.readout);
    return model;
  });
}

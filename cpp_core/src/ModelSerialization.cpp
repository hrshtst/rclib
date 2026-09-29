#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <set>
#include <streambuf>
#include <string>
#include <system_error>
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

// An output stream buffer over a C FILE, so the model streams straight into the
// temporary file without being held in memory.
class FileBuffer : public std::streambuf {
public:
  explicit FileBuffer(std::FILE *file) : file(file) {}

protected:
  std::streamsize xsputn(const char *data, std::streamsize size) override {
    return static_cast<std::streamsize>(std::fwrite(data, 1, static_cast<std::size_t>(size), file));
  }
  int_type overflow(int_type ch) override {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
      return traits_type::not_eof(ch);
    }
    return std::fputc(ch, file) == EOF ? traits_type::eof() : ch;
  }

private:
  std::FILE *file;
};

// Creates a file next to `path` under a random name that did not exist before.
// fopen's "x" mode (C11) fails if anything is at that name, including a planted
// symlink, so the save never writes through a path it does not own. Retries on
// name collisions and throws for any other error.
std::FILE *createTemporaryFile(const std::string &path, std::string &temporary_path) {
  std::random_device random;
  for (int attempt = 0; attempt < 16; ++attempt) {
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), ".%08x%08x.tmp", random(), random());
    temporary_path = path + suffix;
    errno = 0;
    if (std::FILE *file = std::fopen(temporary_path.c_str(), "wbx")) {
      return file;
    }
    if (errno != EEXIST) {
      throw SerializationError("Model: cannot create '" + temporary_path + "': " + std::strerror(errno) + ".");
    }
  }
  throw SerializationError("Model: cannot create a unique temporary file next to '" + path + "'.");
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

// Everything save() checks before writing: the shared checks plus the save-only ones.
void checkSavable(const std::vector<std::shared_ptr<Reservoir>> &reservoirs, const std::string &connection_type,
                  const std::shared_ptr<Readout> &readout) {
  checkModel(reservoirs, connection_type, readout);
  // The format stores each reservoir by value; a shared one would load as two
  // independent objects and change how the model evolves.
  std::set<const Reservoir *> seen;
  for (const auto &reservoir : reservoirs) {
    if (!seen.insert(reservoir.get()).second) {
      throw SerializationError("Model: cannot save a model that holds the same reservoir object more than once.");
    }
  }
}

} // namespace

void Model::save(std::ostream &os) const {
  translateSerializationErrors("Model", [&] {
    checkSavable(reservoirs, connection_type, readout);

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

void Model::save(const std::string &path) const {
  translateSerializationErrors("Model", [&] {
    checkSavable(reservoirs, connection_type, readout); // before touching the filesystem

    std::string temporary_path;
    std::FILE *file = createTemporaryFile(path, temporary_path);
    try {
      FileBuffer buffer(file);
      std::ostream os(&buffer);
      save(os);
      const bool flushed = std::fflush(file) == 0 && std::ferror(file) == 0;
      const bool closed = std::fclose(file) == 0;
      file = nullptr;
      if (!flushed || !closed) {
        throw SerializationError("Model: failed to write '" + temporary_path + "'.");
      }
      std::error_code error;
      std::filesystem::rename(temporary_path, path, error);
      if (error) {
        throw SerializationError("Model: cannot replace '" + path + "': " + error.message() + ".");
      }
    } catch (...) {
      // Cleanup never throws, so the original error is the one reported, and it
      // only removes the file this call created.
      if (file != nullptr) {
        std::fclose(file);
      }
      std::error_code ignored;
      std::filesystem::remove(temporary_path, ignored);
      throw;
    }
  });
}

Model Model::load(const std::string &path) {
  return translateSerializationErrors("Model", [&] {
    errno = 0;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      throw SerializationError("Model: cannot open '" + path + "'" +
                               (errno != 0 ? std::string(": ") + std::strerror(errno) : std::string()) + ".");
    }
    return load(file);
  });
}

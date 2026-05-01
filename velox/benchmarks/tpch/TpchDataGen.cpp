/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Standalone utility that generates TPC-H data at a given scale factor using
// Velox's own data-generation path (TpchConnector) and writes it as Parquet
// files via Velox's writer. Output layout is Hive-style:
//
//   <output_dir>/customer/*.parquet
//   <output_dir>/lineitem/*.parquet
//   <output_dir>/...
//
// The layout matches what `velox_tpch_benchmark --data_path=<output_dir>`
// expects, so the same directory can be consumed both by the benchmark and by
// other engines (e.g., DuckDB) for cross-engine comparison.

#include <gflags/gflags.h>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include <folly/init/Init.h>
#include <glog/logging.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/tpch/gen/TpchGen.h"

DEFINE_string(
    output_dir,
    "",
    "Output directory for generated TPC-H Parquet files. A subdirectory is "
    "created per table.");

DEFINE_double(
    scale_factor,
    1.0,
    "TPC-H scale factor. SF=1 produces ~6M lineitem rows / ~1GB; SF=10 "
    "produces ~60M lineitem rows / ~10GB.");

DEFINE_int32(
    num_splits_per_table,
    8,
    "Number of TpchConnectorSplits per table. Higher values increase scan "
    "parallelism during generation.");

DEFINE_int32(
    num_drivers,
    4,
    "Maximum driver concurrency for the scan-then-write pipeline.");

namespace {

bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}

bool positiveDouble(const char* /*flagName*/, double value) {
  return value > 0.0;
}

bool positiveInt(const char* /*flagName*/, int32_t value) {
  return value > 0;
}

} // namespace

DEFINE_validator(output_dir, &notEmpty);
DEFINE_validator(scale_factor, &positiveDouble);
DEFINE_validator(num_splits_per_table, &positiveInt);
DEFINE_validator(num_drivers, &positiveInt);

namespace {

using namespace facebook::velox;

constexpr std::string_view kTpchConnectorId{"test-tpch"};
constexpr std::string_view kHiveConnectorId{"test-hive"};

// Registers everything needed to scan via the in-memory TpchConnector and
// write Parquet files through the Hive connector.
void registerVeloxComponents() {
  filesystems::registerLocalFileSystem();
  parquet::registerParquetWriterFactory();

  auto emptyConfig = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});

  connector::tpch::TpchConnectorFactory tpchFactory;
  auto tpchConnector =
      tpchFactory.newConnector(std::string(kTpchConnectorId), emptyConfig);
  connector::ConnectorRegistry::global().insert(
      tpchConnector->connectorId(), tpchConnector);

  connector::hive::HiveConnectorFactory hiveFactory;
  auto hiveConnector =
      hiveFactory.newConnector(std::string(kHiveConnectorId), emptyConfig);
  connector::ConnectorRegistry::global().insert(
      hiveConnector->connectorId(), hiveConnector);
}

void unregisterVeloxComponents() {
  connector::ConnectorRegistry::global().erase(std::string(kHiveConnectorId));
  connector::ConnectorRegistry::global().erase(std::string(kTpchConnectorId));
  parquet::unregisterParquetWriterFactory();
}

// Generates a single TPC-H table at the configured scale factor and writes it
// as Parquet under <outputDir>/<tableName>/. Returns the wall-clock duration
// of the scan-then-write pipeline.
std::chrono::milliseconds generateTable(
    tpch::Table table,
    const std::string& outputDir,
    double scaleFactor,
    size_t numSplits,
    int32_t numDrivers,
    memory::MemoryPool* pool) {
  const auto tableName = std::string(tpch::toTableName(table));
  const auto tableDir = fmt::format("{}/{}", outputDir, tableName);
  std::filesystem::create_directories(tableDir);

  auto columnNames = tpch::getTableSchema(table)->names();

  auto plan =
      exec::test::PlanBuilder()
          .tpchTableScan(
              table, std::move(columnNames), scaleFactor, kTpchConnectorId)
          .tableWrite(tableDir, dwio::common::FileFormat::PARQUET)
          .planNode();

  std::vector<exec::Split> splits;
  splits.reserve(numSplits);
  for (size_t i = 0; i < numSplits; ++i) {
    splits.emplace_back(
        std::make_shared<connector::tpch::TpchConnectorSplit>(
            std::string(kTpchConnectorId),
            /*cacheable=*/true,
            /*totalParts=*/numSplits,
            /*partNumber=*/i));
  }

  const auto start = std::chrono::steady_clock::now();
  exec::test::AssertQueryBuilder(plan)
      .maxDrivers(numDrivers)
      .splits(std::move(splits))
      .copyResults(pool);
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  registerVeloxComponents();

  std::filesystem::create_directories(FLAGS_output_dir);

  auto rootPool = memory::memoryManager()->addRootPool("tpch_data_gen");
  auto leafPool = rootPool->addLeafChild("leaf");

  LOG(INFO) << fmt::format(
      "Generating TPC-H scale factor {} into {} (splits={}, drivers={})",
      FLAGS_scale_factor,
      FLAGS_output_dir,
      FLAGS_num_splits_per_table,
      FLAGS_num_drivers);

  const auto totalStart = std::chrono::steady_clock::now();
  for (auto table : tpch::tables) {
    const auto expectedRows = tpch::getRowCount(table, FLAGS_scale_factor);
    const auto elapsed = generateTable(
        table,
        FLAGS_output_dir,
        FLAGS_scale_factor,
        static_cast<size_t>(FLAGS_num_splits_per_table),
        FLAGS_num_drivers,
        leafPool.get());
    LOG(INFO) << fmt::format(
        "  {:>10} : ~{} rows in {} ms",
        tpch::toTableName(table),
        expectedRows,
        elapsed.count());
  }
  const auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - totalStart);
  LOG(INFO) << fmt::format(
      "Done. Total wall time: {} s. Output: {}",
      totalElapsed.count(),
      FLAGS_output_dir);

  unregisterVeloxComponents();
  return 0;
}

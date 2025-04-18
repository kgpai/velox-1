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

#include "velox/vector/fuzzer/ConstrainedVectorGenerator.h"

#include "velox/expression/VectorWriters.h"
#include "velox/vector/fuzzer/Utils.h"

namespace facebook::velox::fuzzer {

using exec::GenericWriter;
using exec::VectorWriter;

// static
VectorPtr ConstrainedVectorGenerator::generateConstant(
    const AbstractInputGeneratorPtr& customGenerator,
    vector_size_t size,
    memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(customGenerator);
  VELOX_CHECK(customGenerator->type()->isPrimitiveType());

  const auto& type = customGenerator->type();
  const auto variant = customGenerator->generate();

  return BaseVector::createConstant(type, variant, size, pool);
}

// static
VectorPtr ConstrainedVectorGenerator::generateFlat(
    const AbstractInputGeneratorPtr& customGenerator,
    vector_size_t size,
    memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(customGenerator);

  VectorPtr result;
  const auto& type = customGenerator->type();
  BaseVector::ensureWritable(SelectivityVector(size), type, pool, result);
  VectorWriter<Any> writer;
  writer.init(*result);

  for (auto i = 0; i < size; ++i) {
    writer.setOffset(i);
    const auto variant = customGenerator->generate();
    if (variant.isNull()) {
      writer.commitNull();
    } else {
      VELOX_DYNAMIC_TYPE_DISPATCH(
          writeOne, type->kind(), variant, writer.current());
      writer.commit(true);
    }
  }
  return result;
}

} // namespace facebook::velox::fuzzer

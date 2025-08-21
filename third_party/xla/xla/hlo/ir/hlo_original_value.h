/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_
#define XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "xla/shape_util.h"
#include "xla/tuple_tree.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// The information of an array in an unoptimized HLO module.
struct OriginalArray {
  // The name of the instruction in the unoptimized HLO module that produces
  // this array or a tuple that includes this array.
  std::string instruction_name;
  // Shape index of the array if the instruction produces a tuple.
  ShapeIndex shape_index;
  std::string ToString() const;
  OriginalArrayProto ToProto() const;
  static OriginalArray FromProto(
      const xla::OriginalArrayProto& original_array_proto);

  friend bool operator==(const OriginalArray& lhs, const OriginalArray& rhs) {
    return lhs.instruction_name == rhs.instruction_name &&
           lhs.shape_index == rhs.shape_index;
  }

  friend bool operator!=(const OriginalArray& lhs, const OriginalArray& rhs) {
    return !(lhs == rhs);
  }

  template <typename H>
  friend H AbslHashValue(H h, const OriginalArray& original_array) {
    return H::combine(std::move(h), original_array.instruction_name,
                      original_array.shape_index);
  }
};

// The information of an HLO value produced by an instruction in an unoptimized
// HLO module.
class OriginalValue {
 public:
  OriginalValue() = default;
  explicit OriginalValue(
      TupleTree<std::optional<OriginalArray>>::Node&& root_node)
      : tree_(std::move(root_node)) {}
  explicit OriginalValue(TupleTree<std::optional<OriginalArray>>&& tree)
      : tree_(std::move(tree)) {}
  std::string ToString() const;
  OriginalValueProto ToProto() const;
  static std::shared_ptr<OriginalValue> FromProto(
      const xla::OriginalValueProto& original_value_proto);
  static std::shared_ptr<OriginalValue> CreateFromInstruction(
      const HloInstruction* instruction, absl::string_view prefix = "");

  const std::optional<OriginalArray>& element(ShapeIndexView index) const {
    return tree_.element(index);
  }
  std::optional<OriginalArray>* mutable_element(ShapeIndexView index) {
    return tree_.mutable_element(index);
  }

  auto elements() const { return tree_.leaves(); }
  auto elements() { return tree_.leaves(); }

  void CopySubtreeFrom(const OriginalValue& other, const ShapeIndex& src_index,
                       const ShapeIndex& dst_index) {
    tree_.CopySubtreeFrom(other.tree_, src_index, dst_index);
  }

  bool operator==(const OriginalValue& other) const {
    auto this_leaves = elements();
    auto other_leaves = other.elements();
    auto this_it = this_leaves.begin();
    auto other_it = other_leaves.begin();

    while (this_it != this_leaves.end() && other_it != other_leaves.end()) {
      if (this_it->first != other_it->first ||
          this_it->second != other_it->second) {
        return false;
      }
      ++this_it;
      ++other_it;
    }
    return this_it == this_leaves.end() && other_it == other_leaves.end();
  }

  bool operator!=(const OriginalValue& other) const {
    return !(*this == other);
  }

  template <typename H>
  friend H AbslHashValue(H h, const OriginalValue& value) {
    for (const auto& leaf : value.elements()) {
      h = H::combine(std::move(h), leaf.first, leaf.second);
    }
    return h;
  }

 private:
  TupleTree<std::optional<OriginalArray>> tree_;
};

struct OriginalValuePointer {
  OriginalValuePointer() = default;
  explicit OriginalValuePointer(
      std::shared_ptr<xla::OriginalValue> original_value) {
    this->original_value = std::move(original_value);
  }

  friend bool operator==(const OriginalValuePointer& lhs,
                         const OriginalValuePointer& rhs) {
    if (!lhs.original_value && !rhs.original_value) {
      return true;
    }
    if (!lhs.original_value || !rhs.original_value) {
      return false;
    }
    return *lhs.original_value == *rhs.original_value;
  }

  template <typename H>
  friend H AbslHashValue(H h, const OriginalValuePointer& value) {
    if (!value.original_value) {
      return H::combine(std::move(h), 0);  // Hash for nullptr
    }
    return AbslHashValue(std::move(h), *value.original_value);
  }

  std::shared_ptr<xla::OriginalValue> original_value = nullptr;
};

// Copies the original value of the source to the destination instruction. This
// performs a deep copy if clone is set to true. Otherwise, it performs a
// shallow copy.
void CopyOriginalValue(const HloInstruction* src_instruction,
                       HloInstruction* dest_instruction, bool clone);

// Removes duplicates of original value objects referenced in the module to save
// memory storage.
void DeduplicateOriginalValues(HloModule* module);
}  // namespace xla

#endif  // XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_

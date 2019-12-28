// Copyright 2018 Slightech Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "util/cam_utils.h"

#include <iostream>

MYNTEYE_BEGIN_NAMESPACE

namespace util {

std::shared_ptr<std::ios> new_format(int width, int prec, char fillch) {
  auto fmt = std::make_shared<std::ios>(nullptr);
  fmt->setf(std::ios::fixed);
  if (width > 0)
    fmt->width(std::move(width));
  if (prec > 0)
    fmt->precision(std::move(prec));
  fmt->fill(std::move(fillch));
  return fmt;
}

}  // namespace util

MYNTEYE_END_NAMESPACE

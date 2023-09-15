/* Copyright 2023 The JAX Authors.

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
#include <Python.h>

#include <string>
#include <string_view>
#include <utility>

#include "nanobind/nanobind.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_priv.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/python/status_casters.h"
#include "xla/status.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {
namespace {
Status RegisterCustomCallTarget(const PJRT_Api* c_api, nb::str fn_name,
                                nb::capsule fn) {
  static const char* const kName = "xla._CUSTOM_CALL_TARGET";
  if (std::string_view(fn.name()) != kName) {
    return InvalidArgument(
        "Argument to RegisterCustomCallTargetRegistry was not a "
        "xla._CUSTOM_CALL_TARGET capsule.");
  }

  if (c_api->priv == nullptr) {
    return Unimplemented("The plugin does not set priv field.");
  }
  auto c_api_priv = static_cast<PJRT_Api_Gpu_Priv*>(c_api->priv);
  TF_RETURN_IF_ERROR(pjrt::CheckMatchingStructSizes(
      "PJRT_Api_Gpu_Priv", PJRT_Api_Gpu_Priv_STRUCT_SIZE,
      c_api_priv->struct_size));

  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = fn_name.c_str();
  args.function_name_size = nb::len(fn_name);
  args.custom_call_function = static_cast<void*>(fn.data());
  RETURN_STATUS_IF_PJRT_ERROR(c_api_priv->custom_call(&args), c_api);
  return OkStatus();
}
}  // namespace

NB_MODULE(gpu_plugin_extension, m) {
  m.def("register_gpu_custom_call_target", [](nb::capsule c_api,
                                              nb::str fn_name, nb::capsule fn,
                                              nb::str xla_platform_name) {
    xla::ThrowIfError(RegisterCustomCallTarget(
        static_cast<const PJRT_Api*>(c_api.data()), fn_name, std::move(fn)));
  });
}
}  // namespace xla

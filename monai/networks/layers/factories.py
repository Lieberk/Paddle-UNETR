# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Defines factories for creating layers in generic, extensible, and dimensionally independent ways. A separate factory
object is created for each type of layer, and factory functions keyed to names are added to these objects. Whenever
a layer is requested the factory name and any necessary arguments are passed to the factory object. The return value
is typically a type but can be any callable producing a layer object.

The factory objects contain functions keyed to names converted to upper case, these names can be referred to as members
of the factory so that they can function as constant identifiers. eg. instance normalization is named `Norm.INSTANCE`.

For example, to get a transpose convolution layer the name is needed and then a dimension argument is provided which is
passed to the factory function:

.. code-block:: python

    dimension = 3
    name = Conv.CONVTRANS
    conv = Conv[name, dimension]

This allows the `dimension` value to be set in the constructor, for example so that the dimensionality of a network is
parameterizable. Not all factories require arguments after the name, the caller must be aware which are required.

Defining new factories involves creating the object then associating it with factory functions:

.. code-block:: python

    fact = LayerFactory()

    @fact.factory_function('test')
    def make_something(x, y):
        # do something with x and y to choose which layer type to return
        return SomeLayerType
    ...

    # request object from factory TEST with 1 and 2 as values for x and y
    layer = fact[fact.TEST, 1, 2]

Typically the caller of a factory would know what arguments to pass (ie. the dimensionality of the requested type) but
can be parameterized with the factory name and the arguments to pass to the created type at instantiation time:

.. code-block:: python

    def use_factory(fact_args):
        fact_name, type_args = split_args
        layer_type = fact[fact_name, 1, 2]
        return layer_type(**type_args)
    ...

    kw_args = {'arg0':0, 'arg1':True}
    layer = use_factory( (fact.TEST, kwargs) )
"""

from typing import Any, Callable, Dict, Tuple, Type, Union

import paddle.nn as nn

from monai.utils import look_up_option

__all__ = ["LayerFactory", "Dropout", "Norm", "Act", "Conv", "Pool", "Pad", "split_args"]


class LayerFactory:
    """
    Factory object for creating layers, this uses given factory functions to actually produce the types or constructing
    callables. These functions are referred to by name and can be added at any time.
    """

    def __init__(self) -> None:
        self.factories: Dict[str, Callable] = {}

    @property
    def names(self) -> Tuple[str, ...]:
        """
        Produces all factory names.
        """

        return tuple(self.factories)

    def add_factory_callable(self, name: str, func: Callable) -> None:
        """
        Add the factory function to this object under the given name.
        """

        self.factories[name.upper()] = func
        self.__doc__ = (
            "The supported member"
            + ("s are: " if len(self.names) > 1 else " is: ")
            + ", ".join(f"``{name}``" for name in self.names)
            + ".\nPlease see :py:class:`monai.networks.layers.split_args` for additional args parsing."
        )

    def factory_function(self, name: str) -> Callable:
        """
        Decorator for adding a factory function with the given name.
        """

        def _add(func: Callable) -> Callable:
            self.add_factory_callable(name, func)
            return func

        return _add

    def get_constructor(self, factory_name: str, *args) -> Any:
        """
        Get the constructor for the given factory name and arguments.

        Raises:
            TypeError: When ``factory_name`` is not a ``str``.

        """

        if not isinstance(factory_name, str):
            raise TypeError(f"factory_name must a str but is {type(factory_name).__name__}.")

        func = look_up_option(factory_name.upper(), self.factories)
        return func(*args)

    def __getitem__(self, args) -> Any:
        """
        Get the given name or name/arguments pair. If `args` is a callable it is assumed to be the constructor
        itself and is returned, otherwise it should be the factory name or a pair containing the name and arguments.
        """

        # `args[0]` is actually a type or constructor
        if callable(args):
            return args

        # `args` is a factory name or a name with arguments
        if isinstance(args, str):
            name_obj, args = args, ()
        else:
            name_obj, *args = args

        return self.get_constructor(name_obj, *args)

    def __getattr__(self, key):
        """
        If `key` is a factory name, return it, otherwise behave as inherited. This allows referring to factory names
        as if they were constants, eg. `Fact.FOO` for a factory Fact with factory function foo.
        """

        if key in self.factories:
            return key

        return super().__getattribute__(key)


def split_args(args):
    """
    Split arguments in a way to be suitable for using with the factory types. If `args` is a string it's interpreted as
    the type name.

    Args:
        args (str or a tuple of object name and kwarg dict): input arguments to be parsed.

    Raises:
        TypeError: When ``args`` type is not in ``Union[str, Tuple[Union[str, Callable], dict]]``.

    """

    if isinstance(args, str):
        return args, {}
    name_obj, name_args = args

    if not isinstance(name_obj, (str, Callable)) or not isinstance(name_args, dict):
        msg = "Layer specifiers must be single strings or pairs of the form (name/object-types, argument dict)"
        raise TypeError(msg)

    return name_obj, name_args


# Define factories for these layer types

Dropout = LayerFactory()
Norm = LayerFactory()
Act = LayerFactory()
Conv = LayerFactory()
Pool = LayerFactory()
Pad = LayerFactory()


@Dropout.factory_function("dropout")
def dropout_factory(dim: int) -> Type[Union[nn.Dropout, nn.Dropout2D, nn.Dropout3D]]:
    types = (nn.Dropout, nn.Dropout2D, nn.Dropout3D)
    return types[dim - 1]


@Dropout.factory_function("alphadropout")
def alpha_dropout_factory(_dim):
    return nn.AlphaDropout


@Norm.factory_function("instance")
def instance_factory(dim: int) -> Type[Union[nn.InstanceNorm1D, nn.InstanceNorm2D, nn.InstanceNorm3D]]:
    types = (nn.InstanceNorm1D, nn.InstanceNorm2D, nn.InstanceNorm3D)
    return types[dim - 1]


@Norm.factory_function("batch")
def batch_factory(dim: int) -> Type[Union[nn.BatchNorm1D, nn.BatchNorm2D, nn.BatchNorm3D]]:
    types = (nn.BatchNorm1D, nn.BatchNorm2D, nn.BatchNorm3D)
    return types[dim - 1]


@Norm.factory_function("group")
def group_factory(_dim) -> Type[nn.GroupNorm]:
    return nn.GroupNorm


@Norm.factory_function("layer")
def layer_factory(_dim) -> Type[nn.LayerNorm]:
    return nn.LayerNorm


@Norm.factory_function("localresponse")
def local_response_factory(_dim) -> Type[nn.LocalResponseNorm]:
    return nn.LocalResponseNorm


@Norm.factory_function("syncbatch")
def sync_batch_factory(_dim) -> Type[nn.SyncBatchNorm]:
    return nn.SyncBatchNorm


Act.add_factory_callable("elu", lambda: nn.ELU)
Act.add_factory_callable("relu", lambda: nn.ReLU)
Act.add_factory_callable("leakyrelu", lambda: nn.LeakyReLU)
Act.add_factory_callable("prelu", lambda: nn.PReLU)
Act.add_factory_callable("relu6", lambda: nn.ReLU6)
Act.add_factory_callable("selu", lambda: nn.SELU)
Act.add_factory_callable("gelu", lambda: nn.GELU)
Act.add_factory_callable("sigmoid", lambda: nn.Sigmoid)
Act.add_factory_callable("tanh", lambda: nn.Tanh)
Act.add_factory_callable("softmax", lambda: nn.Softmax)
Act.add_factory_callable("logsoftmax", lambda: nn.LogSoftmax)


@Conv.factory_function("conv")
def conv_factory(dim: int) -> Type[Union[nn.Conv1D, nn.Conv2D, nn.Conv3D]]:
    types = (nn.Conv1D, nn.Conv2D, nn.Conv3D)
    return types[dim - 1]


@Conv.factory_function("convtrans")
def convtrans_factory(dim: int) -> Type[Union[nn.Conv1DTranspose, nn.Conv2DTranspose, nn.Conv3DTranspose]]:
    types = (nn.Conv1DTranspose, nn.Conv2DTranspose, nn.Conv3DTranspose)
    return types[dim - 1]


@Pool.factory_function("max")
def maxpooling_factory(dim: int) -> Type[Union[nn.MaxPool1D, nn.MaxPool2D, nn.MaxPool3D]]:
    types = (nn.MaxPool1D, nn.MaxPool2D, nn.MaxPool3D)
    return types[dim - 1]


@Pool.factory_function("adaptivemax")
def adaptive_maxpooling_factory(
    dim: int,
) -> Type[Union[nn.AdaptiveMaxPool1D, nn.AdaptiveMaxPool2D, nn.AdaptiveMaxPool3D]]:
    types = (nn.AdaptiveMaxPool1D, nn.AdaptiveMaxPool2D, nn.AdaptiveMaxPool3D)
    return types[dim - 1]


@Pool.factory_function("avg")
def avgpooling_factory(dim: int) -> Type[Union[nn.AvgPool1D, nn.AvgPool2D, nn.AvgPool3D]]:
    types = (nn.AvgPool1D, nn.AvgPool2D, nn.AvgPool3D)
    return types[dim - 1]


@Pool.factory_function("adaptiveavg")
def adaptive_avgpooling_factory(
    dim: int,
) -> Type[Union[nn.AdaptiveAvgPool1D, nn.AdaptiveAvgPool2D, nn.AdaptiveAvgPool3D]]:
    types = (nn.AdaptiveAvgPool1D, nn.AdaptiveAvgPool2D, nn.AdaptiveAvgPool3D)
    return types[dim - 1]
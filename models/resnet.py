import jax
import jax.numpy as jnp
from flax import nnx
from .layers import FilterResponseNorm

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


class LambdaLayer(nnx.Module):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)


class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        option="A",
        norm_type="frn",
        *,
        rngs: nnx.Rngs,
    ):
        self.conv1 = nnx.Conv(
            in_planes,
            planes,
            kernel_size=(3, 3),
            strides=stride,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

        # Choose normalization type
        if norm_type == "bn":
            self.norm1 = nnx.BatchNorm(planes, momentum=0.9, rngs=rngs)
        elif norm_type == "frn":
            self.norm1 = FilterResponseNorm(planes, rngs=rngs)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        self.conv2 = nnx.Conv(
            planes,
            planes,
            kernel_size=(3, 3),
            strides=1,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

        # Choose normalization type for second layer
        if norm_type == "bn":
            self.norm2 = nnx.BatchNorm(planes, momentum=0.9, rngs=rngs)
        elif norm_type == "frn":
            self.norm2 = FilterResponseNorm(planes, rngs=rngs)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        self.shortcut = LambdaLayer(lambda x: x)

        if stride != 1 or in_planes != planes:
            if option == "A":
                # Option A: downsample and zero-pad manually
                self.shortcut = LambdaLayer(
                    lambda x: jnp.pad(
                        x[..., ::2, ::2, :],
                        [(0, 0) * (x.ndim - 3)]
                        + [(0, 0), (0, 0), (planes // 4, planes // 4)],
                        mode="constant",
                        constant_values=0,
                    )
                )
            elif option == "B":
                # Option B: 1x1 conv
                if norm_type == "bn":
                    norm_layer = nnx.BatchNorm(
                        self.expansion * planes, momentum=0.9, rngs=rngs
                    )
                elif norm_type == "frn":
                    norm_layer = FilterResponseNorm(self.expansion * planes, rngs=rngs)
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")

                self.shortcut = nnx.Sequential(
                    nnx.Conv(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=(1, 1),
                        strides=stride,
                        use_bias=False,
                        rngs=rngs,
                    ),
                    norm_layer,
                )

    def __call__(self, x):
        out = nnx.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = nnx.relu(out)
        return out


class ResNet(nnx.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, norm_type="frn", *, rngs: nnx.Rngs
    ):
        self.in_planes = 16
        self.norm_type = norm_type

        self.conv1 = nnx.Conv(
            3,
            16,
            kernel_size=(3, 3),
            strides=1,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

        # Choose normalization type for first layer
        if norm_type == "bn":
            self.norm1 = nnx.BatchNorm(16, momentum=0.9, rngs=rngs)
        elif norm_type == "frn":
            self.norm1 = FilterResponseNorm(16, rngs=rngs)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, rngs=rngs)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, rngs=rngs)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, rngs=rngs)

        self.linear = nnx.Linear(64, num_classes, rngs=rngs)

    def _make_layer(self, block, planes, num_blocks, stride, *, rngs: nnx.Rngs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride=stride,
                    norm_type=self.norm_type,
                    rngs=rngs,
                )
            )
            self.in_planes = planes * block.expansion

        return nnx.Sequential(*layers)

    def __call__(self, x, get_feature=False):
        out = nnx.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nnx.avg_pool(
            out,
            window_shape=(out.shape[-2], out.shape[-2]),
            strides=(out.shape[-2], out.shape[-2]),
        )
        out = out.reshape(x.shape[0], -1)
        feature = out
        out = self.linear(out)
        if get_feature:
            return out, feature
        return out


def resnet20(num_classes, norm_type="frn", rngs: nnx.Rngs = None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    return ResNet(
        BasicBlock, [3, 3, 3], num_classes=num_classes, norm_type=norm_type, rngs=rngs
    )


def resnet32(num_classes, norm_type="frn", rngs: nnx.Rngs = None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    return ResNet(
        BasicBlock, [5, 5, 5], num_classes=num_classes, norm_type=norm_type, rngs=rngs
    )


def resnet44(num_classes, norm_type="frn", rngs: nnx.Rngs = None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    return ResNet(
        BasicBlock, [7, 7, 7], num_classes=num_classes, norm_type=norm_type, rngs=rngs
    )


def resnet56(num_classes, norm_type="frn", rngs: nnx.Rngs = None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    return ResNet(
        BasicBlock, [9, 9, 9], num_classes=num_classes, norm_type=norm_type, rngs=rngs
    )


def resnet110(num_classes, norm_type="frn", rngs: nnx.Rngs = None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    return ResNet(
        BasicBlock,
        [18, 18, 18],
        num_classes=num_classes,
        norm_type=norm_type,
        rngs=rngs,
    )


def resnet1202(num_classes, norm_type="frn", rngs: nnx.Rngs = None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    return ResNet(
        BasicBlock,
        [200, 200, 200],
        num_classes=num_classes,
        norm_type=norm_type,
        rngs=rngs,
    )


def test(net):
    nnx.display(net)


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(f"{net_name} with BatchNorm:")
            test(globals()[net_name](norm_type="bn", rngs=nnx.Rngs(0)))
            print(f"{net_name} with FilterResponseNorm:")
            test(globals()[net_name](norm_type="frn", rngs=nnx.Rngs(0)))
            print()

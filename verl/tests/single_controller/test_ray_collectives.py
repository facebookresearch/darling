import ray
import ray.util.collective as collective
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


@ray.remote
class Actor(Worker):
    @register(Dispatch.ONE_TO_ALL)
    def init(self):
        remote_rank = self.rank // 2
        self.group_name = f"A{self.rank}_R{remote_rank}"
        collective.init_collective_group(world_size=2, rank=0, backend="nccl", group_name=self.group_name)

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def send_tensors(self):
        tensor = torch.ones(size=(4,), dtype=torch.float32, device="cuda") * self.rank
        collective.send(tensor=tensor, dst_rank=1, group_name=self.group_name)


@ray.remote
class Rollout(Worker):
    @register(Dispatch.ONE_TO_ALL)
    def init(self):
        self.remote_first_rank = self.rank * 2
        self.remote_second_rank = self.remote_first_rank + 1
        self.first_group_name = f"A{self.remote_first_rank}_R{self.rank}"
        self.second_group_name = f"A{self.remote_second_rank}_R{self.rank}"

        collective.init_collective_group(world_size=2, rank=1, backend="nccl", group_name=self.first_group_name)
        collective.init_collective_group(world_size=2, rank=1, backend="nccl", group_name=self.second_group_name)

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def receive_tensors(self):
        self.tensor1 = torch.randn(size=(4,), dtype=torch.float32, device="cuda")
        self.tensor2 = torch.randn(size=(4,), dtype=torch.float32, device="cuda")

        collective.recv(self.tensor1, src_rank=0, group_name=self.first_group_name)
        collective.recv(self.tensor2, src_rank=0, group_name=self.second_group_name)

    @register(Dispatch.ONE_TO_ALL)
    def get_tensors(self):
        return {f"src_{self.remote_first_rank}": self.tensor1, f"src_{self.remote_second_rank}": self.tensor2}


def test_ray_collective_group():
    ray.init()

    actor_resource_pool = RayResourcePool([4])
    rollout_resource_pool = RayResourcePool([2])

    actor_cls = RayClassWithInitArgs(cls=Actor)
    rollout_cls = RayClassWithInitArgs(cls=Rollout)

    actor_wg = RayWorkerGroup(
        resource_pool=actor_resource_pool, ray_cls_with_init=actor_cls, name_prefix="collective_group_actor"
    )
    rollout_wg = RayWorkerGroup(
        resource_pool=rollout_resource_pool, ray_cls_with_init=rollout_cls, name_prefix="collective_group_rollout"
    )

    actor_wg.init()
    rollout_wg.init()

    out1 = actor_wg.send_tensors()
    out2 = rollout_wg.receive_tensors()

    # block to wait
    ray.get(out1)
    ray.get(out2)

    output = rollout_wg.get_tensors()

    rollout_0_output = output[0]
    rollout_1_output = output[1]

    output = rollout_0_output | rollout_1_output

    print(output)

    for i in range(4):
        assert torch.sum(output[f"src_{i}"]).item() == 4 * i

    ray.shutdown()


if __name__ == "__main__":
    test_ray_collective_group()

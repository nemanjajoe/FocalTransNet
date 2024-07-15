from config import config_synapse as hyper
from thop import clever_format,profile
import torch



def calc_computation_cost(model,device = torch.device("cuda:0")):
    net = model.to(device)  # 定义好的网络模型
    input = torch.randn(1, 1, 224, 224).cuda()
    flops, params = profile(net, (input,))
    print('flops: ', flops, 'params: ', params)

    iterations = 300  # 重复计算的轮次
    # random_input = torch.randn(1, 3, 224, 224).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(50):
        _ = net(input)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = net(input)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

def print_model_size(model):
    x = torch.randn((1,1,224,224)).to(hyper['device'])
    flops,params = profile(model=model, inputs=(x,)) # type: ignore
    flops,params = clever_format([flops,params], "%.3f")
    print(flops,params)

model = hyper['model'](**hyper['model_args']).to(hyper['device'])
calc_computation_cost(model)


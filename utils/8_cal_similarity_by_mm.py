import torch

g = torch.randn((2,768))
l = torch.randn((2, 256, 768))
print(l)

s = torch.matmul(g, l.permute((0, 2, 1)))
order_q = torch.argsort(s, dim=2, descending=True).squeeze()
print(order_q)

selcted = order_q[:, :64].unsqueeze(2).repeat(1, 1, 768)
select_q = torch.gather(input=l, index=selcted, dim=1)

print(select_q[0])
print(select_q[1])


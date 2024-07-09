
# IIM
    feature, skip1, skip2, skip3, proto_CT1, proto1, ptr1, coarse_pred1 = model.data_encoder(image1, mask1)
    output = model.data_decoder(feature, skip1, skip2, skip3)

    feature2, skip21, skip22, skip23, proto_CT2, _, ptr2, _ = model.data_encoder(image2, mask2, mode='aug')
    output2 = model.data_decoder(feature2, skip21, skip22, skip23)
    loss_dsc = criterion_dsc(torch.softmax(output, dim=1), mask_multi1) + criterion_dsc(torch.softmax(output2, dim=1),
                                                                                        mask_multi2)
    loss_bce = criterion_bce(output, mask1[:, 0, :, :, :].long()) + criterion_bce(output2, mask2[:, 0, :, :, :].long())
    loss_aux = criterion_dsc(coarse_pred1, mask_multi1)
    proto_CT = torch.cat((proto_CT1, proto_CT2), dim=0)
    proto_ptr = torch.cat((ptr1, ptr2), dim=0)
    proto_CT, proto_ptr = proto_nosingle(proto_CT, proto_ptr)
    loss_CT = criterion_CT(proto_CT, proto_ptr)
    loss_CT_list.append(loss_CT.item())

    loss_aux_list.append(loss_aux.item())
    loss = loss_dsc + loss_bce + loss_aux + loss_CT

# FIM
    feature3 = feature.clone()

    new_proto = []
    for j in ptr1:
        rand = random.randint(0, 49)
        temp_style = proto_list[j][rand]
        new_proto.append(temp_style.unsqueeze(0))
    new_proto = torch.cat(new_proto, dim=0)

    B, C, D, H, W = feature3.shape
    coarse_pred = torch.nn.Upsample(scale_factor=1 / 8, mode='trilinear')(coarse_pred1)
    coarse_pred = coarse_pred.view(B, -1, D * H * W)  # B, n_classes, N

    coarse_pred_filter = []
    for j in range(len(ptr1)):
        idx = ptr1[j].item()
        coarse_pred_filter.append(coarse_pred[:, idx])
    coarse_pred_filter = torch.stack(coarse_pred_filter, dim=1)
    delta = torch.bmm(coarse_pred_filter.permute(0, 2, 1), (new_proto - proto1).unsqueeze(0))  # B, N, C
    delta = delta.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

    delta_lambda = 0.05 * (i // (0.05 * 300) + 0.5)
    z = torch.tensor(random.uniform(0.1, 0.1 + delta_lambda))
    z = torch.clamp(z, 0.1, 0.5)
    feature3 = feature3 + delta * z
    output3 = model.data_decoder(feature3, skip1, skip2, skip3)

# save_style():
    for j in range(len(ptr1)):
        idx = ptr1[j].item()
        if len(proto_list[idx]) >= 50:
            proto_list[idx] = proto_list[idx][1:]
        proto_list[idx].append(proto1[j].detach())


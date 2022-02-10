import torch
import copy

def prune_segmenter_layers(ws, layer_list=[0]):
    ws.segmenter_model.model.scorer.model = \
        deleteEncodingLayers(ws.segmenter_model.model.scorer.model, layer_list=layer_list)
    return ws

def deleteEncodingLayers(model, layer_list=[0]):
    oldModuleList = model.transformer.h
    newModuleList = torch.nn.ModuleList()

    for index in layer_list:
        newModuleList.append(oldModuleList[index])

    copyOfModel = copy.deepcopy(model)
    copyOfModel.transformer.h = newModuleList

    return copyOfModel
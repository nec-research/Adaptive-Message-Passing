#!/bin/bash

pydgn-dataset --config-file DATA_CONFIGS/config_PeptidesFunc.yml
pydgn-dataset --config-file DATA_CONFIGS/config_PeptidesStruct.yml

pydgn-train --config-file LRGB_CONFIGS/config_UDN_GCN_PeptidesFunc.yml
pydgn-train --config-file LRGB_CONFIGS/config_UDN_GCN_PeptidesStruct.yml

pydgn-train --config-file LRGB_CONFIGS/config_UDN_GINE_PeptidesFunc.yml
pydgn-train --config-file LRGB_CONFIGS/config_UDN_GINE_PeptidesStruct.yml

pydgn-train --config-file LRGB_CONFIGS/config_UDN_GatedGCN_PeptidesFunc.yml
pydgn-train --config-file LRGB_CONFIGS/config_UDN_GatedGCN_PeptidesStruct.yml

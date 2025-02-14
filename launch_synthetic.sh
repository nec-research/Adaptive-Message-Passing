#!/bin/bash

pydgn-dataset --config-file DATA_CONFIGS/config_Diameter.yml
pydgn-dataset --config-file DATA_CONFIGS/config_Eccentricity.yml
pydgn-dataset --config-file DATA_CONFIGS/config_SSSP.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_ADGN_ECC.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_ADGN_ECC.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_ADGN_SSSP.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_ADGN_SSSP.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_ADGN_Diameter.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_ADGN_Diameter.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GCN_ECC.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_GCN_ECC.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GCN_SSSP.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_GCN_SSSP.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GCN_Diameter.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_GCN_Diameter.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GIN_ECC.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_GIN_ECC.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GIN_SSSP.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_GIN_SSSP.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GIN_Diameter.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_UDN_GIN_Diameter.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GAT_ECC.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_GAT_SSSP.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_GAT_Diameter.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_SAGE_ECC.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_SAGE_SSSP.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_SAGE_Diameter.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GCN2_ECC.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_GCN2_SSSP.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_GCN2_Diameter.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_GRAND_ECC.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_GRAND_SSSP.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_GRAND_Diameter.yml

pydgn-train --config-file SYNTHETIC_CONFIGS/config_DGC_ECC.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_DGC_SSSP.yml
pydgn-train --config-file SYNTHETIC_CONFIGS/config_DGC_Diameter.yml

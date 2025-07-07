pydgn-dataset --config-file DATA_CONFIGS/config_Chameleon.yml
pydgn-dataset --config-file DATA_CONFIGS/config_Squirrel.yml
pydgn-dataset --config-file DATA_CONFIGS/config_Texas.yml
pydgn-dataset --config-file DATA_CONFIGS/config_Wisconsin.yml
pydgn-dataset --config-file DATA_CONFIGS/config_Actor.yml
pydgn-dataset --config-file DATA_CONFIGS/config_Citeseer.yml
pydgn-dataset --config-file DATA_CONFIGS/config_Cora.yml
pydgn-dataset --config-file DATA_CONFIGS/config_Pubmed.yml


pydgn-train --config-file NODE_CLASS_CONFIGS/config_GCN_Actor.yml
pydgn-train --config-file NODE_CLASS_CONFIGS/config_UDN_GCN_Actor.yml

pydgn-train --config-file NODE_CLASS_CONFIGS/config_GCN_Wisconsin.yml
pydgn-train --config-file NODE_CLASS_CONFIGS/config_UDN_GCN_Wisconsin.yml

pydgn-train --config-file NODE_CLASS_CONFIGS/config_GCN_Chameleon.yml
pydgn-train --config-file NODE_CLASS_CONFIGS/config_UDN_GCN_Chameleon.yml

pydgn-train --config-file NODE_CLASS_CONFIGS/config_GCN_Squirrel.yml
pydgn-train --config-file NODE_CLASS_CONFIGS/config_UDN_GCN_Squirrel.yml

pydgn-train --config-file NODE_CLASS_CONFIGS/config_GCN_Texas.yml
pydgn-train --config-file NODE_CLASS_CONFIGS/config_UDN_GCN_Texas.yml

pydgn-train --config-file NODE_CLASS_CONFIGS/config_GCN_Citeseer.yml
pydgn-train --config-file NODE_CLASS_CONFIGS/config_UDN_GCN_Citeseer.yml

pydgn-train --config-file NODE_CLASS_CONFIGS/config_GCN_Cora.yml
pydgn-train --config-file NODE_CLASS_CONFIGS/config_UDN_GCN_Cora.yml

pydgn-train --config-file NODE_CLASS_CONFIGS/config_GCN_Pubmed.yml
pydgn-train --config-file NODE_CLASS_CONFIGS/config_UDN_GCN_Pubmed.yml
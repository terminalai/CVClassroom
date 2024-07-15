## Setup instructions 

Download model from [https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_stanford_card_96.41.pth] and put in this folder. 

Download the src_files folder from [https://github.com/Alibaba-MIIL/ML_Decoder/tree/main] and put it under this folder. 
Modify src_files/models/utils/factory, rearranging these lines: 
```        if 'model' in state:
            key = 'model'
        else:
            key = 'state_dict'

        if not load_head:
            
            filtered_dict = {k: v for k, v in state[key].items() if
                             (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)```


Adapted from [https://github.com/Alibaba-MIIL/ML_Decoder] 

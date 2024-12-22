from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
import torch
from typing import List, cast

from tqdm import tqdm
from PIL import Image
import os

import spaces

model_name = "vidore/colpali-v1.2"
device = get_torch_device("cuda")

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

class ColpaliManager:

    
    def __init__(self, device = "cuda", model_name = "vidore/colpali-v1.2"):

        print(f"Initializing ColpaliManager with device {device} and model {model_name}")

        # self.device = get_torch_device(device)

        # self.model = ColPali.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        #     device_map=self.device,
        # ).eval()

        # self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

    @spaces.GPU
    def get_images(self, paths: list[str]) -> List[Image.Image]:
        return [Image.open(path) for path in paths]

    @spaces.GPU
    def process_images(self, image_paths:list[str], batch_size=5):

        print(f"Processing {len(image_paths)} image_paths")
        
        images = self.get_images(image_paths)

        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: processor.process_images(x),
        )

        ds: List[torch.Tensor] = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to(device))))
                
        ds_np = [d.float().cpu().numpy() for d in ds]

        return ds_np
    

    @spaces.GPU
    def process_text(self, texts: list[str]):
        print(f"Processing {len(texts)} texts")

        dataloader = DataLoader(
            dataset=ListDataset[str](texts),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: processor.process_queries(x),
        )

        qs: List[torch.Tensor] = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embeddings_query = model(**batch_query)

            qs.extend(list(torch.unbind(embeddings_query.to(device))))

        qs_np = [q.float().cpu().numpy() for q in qs]

        return qs_np
    



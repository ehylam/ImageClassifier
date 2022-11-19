from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url
from time import sleep

dest = 'ragdoll-cat.jpg'

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')


urls = search_images('ragdoll cat', max_images=1)

download_url(urls[0], dest, show_progress=true)


download_url(search_images('snowshoe cat', max_images=1)[0], 'showshoe-cat.jpg', show_progress=true)

searches = 'snowshoe cat','ragdoll cat'
path = Path('snowshoe_or_ragdoll')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)


failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)


learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
learn.show_results()

learn.export('model.pkl')
# tort
Tort: Self-supervised student-teacher ViT training for mortals

Run:
```
git config --global user.email nds-gfk@mail.ru
git config --global user.name detkov
```

If using torchvision==0.12.0, in `OxfordIIITPet` change 
```
        if self.transforms:
            image, target = self.transforms(image, target)
```
to
```
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

```
If using torchvision==0.12.0, in `Flowers102` change 
```
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))
```
to
```
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))
```

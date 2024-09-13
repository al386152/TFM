from torchvision.datasets import ImageFolder
from collections import Counter

class ModifiedImageFolder(ImageFolder):
    
    # transform: transformación a hacer de normal (con aumento de datos)
    # basic_transform: transformaciones básicas que se tienen que hacer para que vaya (pero sin aumento de datos)
    # classes_not_augment: lista de clases 
    def __init__(self, root:str, transform, 
                 basic_transform, classes_not_augment:list=[]):
        
        super().__init__(root, transform=transform) 
        self.transform = transform
        self.basic_transform = basic_transform
        self.classes_not_augment = classes_not_augment

        #print(f"self.classes: {self.classes}")
        #print(f"self.class_to_idx: {self.class_to_idx}")            

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)       
        
        #print(f"target: {target}")
        #if str(target) in self.classes_not_augment:
        #    print(f"==> TEST[Borrar] {target} in self.classes_not_augment ({self.classes_not_augment})")
        #    image = self.basic_transform(image)  
        #else:
        #    print(f"==> TEST[Borrar] {target} not in self.classes_not_augment ({self.classes_not_augment})")
        #    image = self.transform(image)    

        image = self.basic_transform(image) if target in self.classes_not_augment else self.transform(image)

        return image, target

    def get_proporcion_clase(self) -> dict:
        counter_clases = Counter(self.targets)
        num_imagenes = len(self.samples)
        proporcion = {clase: (numero / num_imagenes) for clase, numero in counter_clases.items()}
        return proporcion
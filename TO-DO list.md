1. Create model and show evaluation metrics 
2. EDA on the current dataset
3. Preprocessing image
4. Augmentation image
5. Train model (method and param, not running)
6. Program clicking logic
7. Design UI
8. Demo display information 

1. Auto crawl - save dir
2. eda - gemini 
3. sample img & entire dataset - preprocess
4. sample img & entire dataset - augmentation
5. training info - evaluation metrics (plot graph)

-----
EDA
-----
Hue and Saturation - preprocess grayscale - morphology (fill the faded shape)
Laplacian - Blur (Noise) - Denoising method






<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAABmJLR0QA/wD/AP+gvaeTAAAC00lEQVR4nO2aTU8TQRyHn39bIdXEm3jwLQhefPkAJorYLYslIF64ohwM8eQH0A/gzYSLIRooxBORKJr4Ultq4smz8YgQb3ow4YAmUHY8IEpgd7vQ3e0smee4+5/uPL+daXdmCwaDwWAwGAwGg8FgMBgM+wBr0u7JFe17QWrTUXcmbqxJuwdhTpDejsHO7Ne5hbJf/b4KYFMeJAuAcKleCPsmgB3ymwiX2m901BZfLHx0a5eKpXcR4ykPgPqdEvnk1Vai7Fgc1JMXkevlm+88p0CiA2hUHhIcQBjykNAAwpKHBAYQpjwkLICw5SFBAUQhDwkJICp5SEAAUcqD5gFELQ8aBxCHPGgaQFzyoGEAccpDwNXgxZmhLCr6sPJTvXk/eRSDYcpDgAAGxgcOZleW31hF+1GUIViTdo9S6qXfna+MlN6HfV3fAApjhdZfrauzInIFkdGoQoh72G/FM4ChmaGW1cPOM+Dav4MRhNBMefAJ4OfK8hjQv+OEyKhV7H0YRgjNmPPb8QxgndQDYMn1pHC30ZHQrDm/Hc8APoy8XVK1dDew6FrQwHTIFe0uRJ43a9hvpW7nc0/6TklmvQq0uxYoNV65VbqDoIJcMFe0uwR5DRxy+bBY5SHgg1B+On9SOZkqqNOuBQFD0E0edvEkuBFCeh7ocC2oE4KO8rCL9wLl4fK3tKOuAguuBT7fCbrKwx7WAvaEfcJJybyCTteCbSNBZ3nY42Ko+2nheKbmVOuFkJuyL+ssDw2sBnNT/cdErVWBMx4ls6D6/B5y4vidr0dDT3PWY+soBzLzwNngrfS485s09HK0crvynbVaDvgSrIVe8hDShsjfkVABznlX6ScPIe4I2dN2W82RisD5nWf1lIeQt8Tsabtt3aEMcuH/UX3lIeQ/SJSGSz9anLQF6vPGEb3lIaJN0cJE4ciaOK9IcV9n+WiJYRPVYDAYDAaDoRH+ALzfixyrasnFAAAAAElFTkSuQmCC" alt="Check mark" style="width: 30px; height: 30px; display: block;">



**xpath - alt="Check mark" -> successful**



\*\*Validation logic instruction\*\*

Now, there is difference validation logic for each challenge type. Write a validation logic for the challenge type with the challenge\_type\_id of ct-002.



For this challenge type, validate the inferenced output classes, if there is a class exist more than 1 times in the inferenced output then the particular object need to be click.  If the class only exist 1 times then that particular object is not selected to be clicked.





For example,

Find the species that has longest leg.

Tiles image consists of cat, birds. (Cat has longest leg than birds - cats will be the answer)



Click on the pocket-sized objects

Tiles images



\*\*Sample image\*\*

Inference class used for question validation logic. 
If there div image and canvas image exists together, div image is inference for the question purpose while the canvas image is inference for the clicking purpose.

Save them in 1 challenges.   



Find objects commonly found in this **setting**. (Kitchen)



Select everything that you can put safely on top of the object **shown**. (table)


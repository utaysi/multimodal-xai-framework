class GradCAMExplainer:
    def __init__(self, model):
        self.model = model
        self.gradcam = GradCAM(model=model)
    
    def explain(self, image):
        # Generate GradCAM heatmap
        return heatmap
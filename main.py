import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        self.window_size = (64, 64)
        
    def blur_image(self, img, blur_size=5):
        """Smooths image to reduce noise and detail."""
        return cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    
    def get_edge_map(self, img):
        """Extracts edges using Canny algorithm."""
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.Canny(grayscale, 40, 120)
    
    def locate_contours(self, edges):
        """Identifies contours from the edges."""
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect_round_shapes(self, img):
        """Detects circular shapes via Hough Transform."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                                   param1=45, param2=25, minRadius=8, maxRadius=80)
        return circles

    def calculate_hog(self, img):
        """Computes HOG (Histogram of Oriented Gradients) for the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.resize(gray, self.window_size)
        
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi % 180
        
        cell_dim = 16
        bins = 9
        histograms = []
        
        for y in range(0, gray.shape[0], cell_dim):
            for x in range(0, gray.shape[1], cell_dim):
                mag = magnitude[y:y+cell_dim, x:x+cell_dim]
                ang = angle[y:y+cell_dim, x:x+cell_dim]
                
                hist = np.zeros(bins)
                indices = (ang / 20).astype(int)
                for bin_idx in range(bins):
                    hist[bin_idx] = np.sum(mag[indices == bin_idx])
                
                histograms.extend(hist)
        
        return np.array(histograms)

    def match_template(self, search_img, template_img, similarity_threshold=0.55):
        print("Starting Detection Sequence...")

        results = {}

        print("- Blurring the images...")
        search_blur = self.blur_image(search_img)
        template_blur = self.blur_image(template_img)
        results['blurred'] = search_blur
        
        print("- Performing edge detection...")
        search_edge_map = self.get_edge_map(search_blur)
        template_edge_map = self.get_edge_map(template_blur)
        results['edges'] = search_edge_map
        
        print("- Detecting contours...")
        search_contours = self.locate_contours(search_edge_map)
        template_contours = self.locate_contours(template_edge_map)
        results['contours'] = search_contours
        
        print("- Searching for circular patterns...")
        found_circles = self.detect_round_shapes(search_blur)
        results['circles'] = found_circles
        
        print("- Calculating HOG descriptors...")
        template_hog = self.calculate_hog(template_img)
        
        top_match = None
        top_score = similarity_threshold
        
        print("- Initiating multi-scale search...")
        for scaling in np.linspace(0.6, 1.4, 7):
            resized_w = int(search_img.shape[1] * scaling)
            resized_h = int(search_img.shape[0] * scaling)
            scaled_img = cv2.resize(search_img, (resized_w, resized_h))
            
            shift = 16
            for j in range(0, resized_h - self.window_size[1], shift):
                for i in range(0, resized_w - self.window_size[0], shift):
                    roi = scaled_img[j:j+self.window_size[1], i:i+self.window_size[0]]
                    
                    if roi.shape[:2] != self.window_size:
                        continue
                    
                    hog_features = self.calculate_hog(roi)
                    score = np.dot(template_hog, hog_features) / (
                        np.linalg.norm(template_hog) * np.linalg.norm(hog_features))
                    
                    if score > top_score:
                        top_score = score
                        top_match = (int(i / scaling), int(j / scaling),
                                     int(self.window_size[0] / scaling),
                                     int(self.window_size[1] / scaling), score)
        
        return top_match, results

def main():
    print("Loading images for analysis...")
    template_img = cv2.imread('images/wanted.png')
    search_img = cv2.imread('images/crowd.jpg')
    
    if search_img is None or template_img is None:
        print("Failed to load images.")
        return
        
    print(f"Processing search image of dimensions: {search_img.shape}")
    print(f"Processing template image of dimensions: {template_img.shape}")
    
    detector = ObjectDetector()
    match, intermediate_steps = detector.match_template(search_img, template_img)
    
    cv2.imshow('Template Preview', template_img)
    cv2.imshow('Blurred Search Image', intermediate_steps['blurred'])
    cv2.imshow('Edge Map', intermediate_steps['edges'])
    
    contour_display = np.zeros_like(search_img)
    cv2.drawContours(contour_display, intermediate_steps['contours'], -1, (0, 255, 255), 2)
    cv2.imshow('Contour Detection', contour_display)
    
    circle_display = search_img.copy()
    if intermediate_steps['circles'] is not None:
        for circle in intermediate_steps['circles'][0]:
            x, y, r = circle
            cv2.circle(circle_display, (int(x), int(y)), int(r), (255, 128, 0), 2)
    cv2.imshow('Circle Highlights', circle_display)
    
    final_result = search_img.copy()
    if match:
        x, y, w, h, confidence = match
        cv2.rectangle(final_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(final_result, f"Match Found: {confidence*100:.0f}%", (x, y - 15),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2)
    else:
        cv2.putText(final_result, "No Suitable Match Found", (10, 40),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Detection Result', final_result)
    
    print("Detection sequence complete. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

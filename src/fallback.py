class HeuristicModel:
    def predict(self, data):
        # Handle missing data
        if not data:
            return {
                "prediction": 0,
                "probability": 0.0,
                "status": "fallback_heuristic"
            }
            
        # Extract key features with defaults
        progress = data.get("Progress_Percentage", 0)
        quiz_score = data.get("Quiz_Score_Avg", 0)
        
        # Case 1: High engagement -> likely to complete
        if progress >= 90 and quiz_score >= 70:
            probability = 0.95
            prediction = 1
            
        # Case 2: Low engagement -> unlikely to complete
        elif progress <= 30:
            probability = 0.1
            prediction = 0
            
        # Case 3: Mixed/Average -> use weighted score
        else:
            # Weight progress 70%, quiz 30%
            score = (progress * 0.7 + quiz_score * 0.3) / 100
            probability = score
            prediction = 1 if probability >= 0.5 else 0
            
        return {
            "prediction": prediction,
            "probability": probability,
            "status": "success"
        }

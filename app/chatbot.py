class ContextAwareChatbot:
    def __init__(self):
        self.context = {}
        self.features = ['school_year', 'age', 'gender', 'height', 'weight', 'phq_score', 'gad_score', 'epworth_score']
        self.current_feature_index = 0

    def get_next_question(self):
        if self.current_feature_index < len(self.features):
            feature = self.features[self.current_feature_index]
            return self.get_question_for_feature(feature)
        return None

    def get_question_for_feature(self, feature):
        questions = {
            'school_year': "What year are you in school? (1-6)",
            'age': "How old are you?",
            'gender': "What is your gender? (male/female/other)",
            'height': "What is your height in cm?",
            'weight': "What is your weight in kg?",
            'phq_score': "On a scale of 0-27, how would you rate your level of depression?",
            'gad_score': "On a scale of 0-21, how would you rate your level of anxiety?",
            'epworth_score': "On a scale of 0-24, how likely are you to doze off during the day?"
        }
        return questions.get(feature, "Invalid feature")

    def process_user_input(self, user_input):
        if self.current_feature_index < len(self.features):
            feature = self.features[self.current_feature_index]
            self.context[feature] = user_input.strip()
            self.current_feature_index += 1

    def is_conversation_complete(self):
        return self.current_feature_index >= len(self.features)

    def get_features(self):
        return self.context


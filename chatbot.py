import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class ContextAwareChatbot:
    def __init__(self):
        self.context = {}
        self.features = ['school_year', 'age', 'gender', 'height', 'weight', 'phq_score', 'gad_score', 'epworth_score']
        self.current_feature = None

    def extract_entities(self, user_input):
        tokens = word_tokenize(user_input.lower())
        tagged = pos_tag(tokens)
        entities = {}
        for word, tag in tagged:
            if tag == 'CD' and 'age' not in entities:
                entities['age'] = word
            elif word in ['male', 'female', 'other'] and 'gender' not in entities:
                entities['gender'] = word
            elif tag == 'CD' and 'height' not in entities:
                entities['height'] = word
            elif tag == 'CD' and 'weight' not in entities:
                entities['weight'] = word
        return entities

    def get_next_question(self):
        for feature in self.features:
            if feature not in self.context:
                self.current_feature = feature
                if feature == 'school_year':
                    return "What year are you in school? (1-6)"
                elif feature == 'age':
                    return "How old are you?"
                elif feature == 'gender':
                    return "What is your gender? (male/female/other)"
                elif feature == 'height':
                    return "What is your height in cm?"
                elif feature == 'weight':
                    return "What is your weight in kg?"
                elif feature == 'phq_score':
                    return "On a scale of 0-27, how would you rate your level of depression?"
                elif feature == 'gad_score':
                    return "On a scale of 0-21, how would you rate your level of anxiety?"
                elif feature == 'epworth_score':
                    return "On a scale of 0-24, how likely are you to doze off during the day?"
        return None

    def process_user_input(self, user_input):
        entities = self.extract_entities(user_input)
        
        if self.current_feature == 'school_year' and user_input.isdigit() and 1 <= int(user_input) <= 6:
            self.context['school_year'] = int(user_input)
        elif self.current_feature == 'age' and user_input.isdigit():
            self.context['age'] = int(user_input)
        elif self.current_feature == 'gender' and user_input.lower() in ['male', 'female', 'other']:
            self.context['gender'] = user_input.lower()
        elif self.current_feature == 'height' and user_input.replace('cm', '').strip().isdigit():
            self.context['height'] = int(user_input.replace('cm', '').strip())
        elif self.current_feature == 'weight' and user_input.replace('kg', '').strip().isdigit():
            self.context['weight'] = int(user_input.replace('kg', '').strip())
        elif self.current_feature in ['phq_score', 'gad_score', 'epworth_score'] and user_input.isdigit():
            self.context[self.current_feature] = int(user_input)
        else:
            return f"I'm sorry, I didn't understand that. Could you please provide a valid answer for {self.current_feature}?"

        next_question = self.get_next_question()
        if next_question:
            return next_question
        else:
            return "Thank you for providing all the information. Is there anything else you'd like to share?"

    def start_conversation(self):
        print("Hello! I'd like to ask you a few questions about your health and well-being.")
        while True:
            question = self.get_next_question()
            if question:
                print(question)
                user_input = input()
                response = self.process_user_input(user_input)
                print(response)
            else:
                print("Thank you for your time. Here's a summary of the information you provided:")
                for feature, value in self.context.items():
                    print(f"{feature}: {value}")
                break

if __name__ == "__main__":
    chatbot = ContextAwareChatbot()
    chatbot.start_conversation()


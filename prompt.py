from typing import Dict, List, Optional
from datetime import datetime
import json
from enum import Enum


class TherapyType(Enum):
    """Enum for therapy types"""
    CBT = "Cognitive Behavioral Therapy"
    DBT = "Dialectical Behavior Therapy"
    ACT = "Acceptance and Commitment Therapy"
    GRIEF = "Grief Counseling"
    ANXIETY = "Anxiety Management"
    PARENTING = "Parenting Support"
    DEPRESSION = "Depression Support"
    TRAUMA = "Trauma-Informed Therapy"
    GENERAL = "General Therapy"


class ConversationStyle(Enum):
    """Enum for conversation styles"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    EMPATHETIC = "empathetic"
    MOTIVATIONAL = "motivational"
    GENTLE = "gentle"


class PromptManager:
    """
    Manages therapy prompts with context from PDFs and therapy-specific guidelines.
    Creates role-based system prompts that guide AI responses.
    """
    
    def __init__(self, 
                 default_therapy_type: TherapyType = TherapyType.GENERAL,
                 conversation_style: ConversationStyle = ConversationStyle.EMPATHETIC):
        """
        Initialize Prompt Manager
        
        Args:
            default_therapy_type: Default therapy approach
            conversation_style: Default conversation style
        """
        self.default_therapy_type = default_therapy_type
        self.conversation_style = conversation_style
        self.session_context = {}
        
        # Therapy-specific guidelines
        self.therapy_guidelines = {
            TherapyType.CBT: {
                "focus": "thought patterns and behaviors",
                "techniques": [
                    "Identify negative thought patterns",
                    "Challenge cognitive distortions",
                    "Develop coping strategies",
                    "Set behavioral goals"
                ],
                "approach": "structured and goal-oriented"
            },
            TherapyType.DBT: {
                "focus": "emotional regulation and mindfulness",
                "techniques": [
                    "Mindfulness exercises",
                    "Distress tolerance skills",
                    "Emotion regulation strategies",
                    "Interpersonal effectiveness"
                ],
                "approach": "validation-based and skill-building"
            },
            TherapyType.ACT: {
                "focus": "psychological flexibility and values",
                "techniques": [
                    "Acceptance of thoughts and feelings",
                    "Mindfulness and present-moment awareness",
                    "Values clarification",
                    "Committed action planning"
                ],
                "approach": "experiential and values-driven"
            },
            TherapyType.GRIEF: {
                "focus": "processing loss and bereavement",
                "techniques": [
                    "Normalize grief reactions",
                    "Honor memories",
                    "Build coping resources",
                    "Navigate grief stages"
                ],
                "approach": "compassionate and patient"
            },
            TherapyType.ANXIETY: {
                "focus": "managing worry and fear",
                "techniques": [
                    "Relaxation techniques",
                    "Grounding exercises",
                    "Exposure therapy concepts",
                    "Worry management strategies"
                ],
                "approach": "calming and reassuring"
            },
            TherapyType.PARENTING: {
                "focus": "family dynamics and child development",
                "techniques": [
                    "Positive parenting strategies",
                    "Communication skills",
                    "Boundary setting",
                    "Developmental understanding"
                ],
                "approach": "supportive and educational"
            },
            TherapyType.DEPRESSION: {
                "focus": "mood improvement and activity",
                "techniques": [
                    "Behavioral activation",
                    "Pleasant activity scheduling",
                    "Cognitive restructuring",
                    "Self-compassion practices"
                ],
                "approach": "encouraging and hopeful"
            },
            TherapyType.TRAUMA: {
                "focus": "safety and stabilization",
                "techniques": [
                    "Grounding techniques",
                    "Safety planning",
                    "Resource building",
                    "Gentle processing"
                ],
                "approach": "trauma-informed and safe"
            },
            TherapyType.GENERAL: {
                "focus": "overall wellbeing",
                "techniques": [
                    "Active listening",
                    "Empathetic support",
                    "Problem-solving",
                    "Resource identification"
                ],
                "approach": "flexible and person-centered"
            }
        }
        
        # Conversation style templates
        self.style_templates = {
            ConversationStyle.PROFESSIONAL: {
                "tone": "professional yet warm",
                "language": "clear and clinical when appropriate",
                "structure": "organized and systematic"
            },
            ConversationStyle.FRIENDLY: {
                "tone": "warm and approachable",
                "language": "conversational and relatable",
                "structure": "flowing and natural"
            },
            ConversationStyle.EMPATHETIC: {
                "tone": "deeply understanding and compassionate",
                "language": "validating and reflective",
                "structure": "responsive to emotional needs"
            },
            ConversationStyle.MOTIVATIONAL: {
                "tone": "encouraging and inspiring",
                "language": "positive and action-oriented",
                "structure": "goal-focused and energizing"
            },
            ConversationStyle.GENTLE: {
                "tone": "soft and nurturing",
                "language": "careful and considerate",
                "structure": "paced and patient"
            }
        }
    
    def set_session_context(self, context: Dict):
        """Update session context with user information"""
        self.session_context.update(context)
    
    def generate_system_prompt(self, 
                             therapy_type: Optional[TherapyType] = None,
                             pdf_context: str = "",
                             user_history: List[Dict] = None) -> str:
        """
        Generate comprehensive system prompt for the AI
        
        Args:
            therapy_type: Specific therapy type for this interaction
            pdf_context: Relevant context from PDF documents
            user_history: Previous conversation history
            
        Returns:
            Complete system prompt
        """
        therapy_type = therapy_type or self.default_therapy_type
        guidelines = self.therapy_guidelines[therapy_type]
        style = self.style_templates[self.conversation_style]
        
        # Build the system prompt
        system_prompt = f"""You are an experienced, compassionate AI therapist specializing in {therapy_type.value}. 
Your approach is {guidelines['approach']}, with a {style['tone']} communication style.

CORE THERAPEUTIC PRINCIPLES:
1. Always maintain professional boundaries while being warm and supportive
2. Never diagnose or prescribe medication
3. Encourage professional help when appropriate
4. Validate feelings while gently challenging unhelpful patterns
5. Focus on {guidelines['focus']}
6. Use {style['language']} language with {style['structure']} responses

THERAPEUTIC TECHNIQUES TO EMPLOY:
{chr(10).join(f"- {technique}" for technique in guidelines['techniques'])}

RESPONSE GUIDELINES:
- Begin responses with validation or acknowledgment
- Ask one thoughtful question at a time
- Offer practical strategies when appropriate
- Keep responses concise but meaningful (2-3 paragraphs typically)
- Use "I" statements and avoid being prescriptive
- Remember this is supportive guidance, not medical advice

"""
        
        # Add PDF context if available
        if pdf_context:
            system_prompt += f"""
RELEVANT THERAPEUTIC KNOWLEDGE:
{pdf_context}

Apply this knowledge naturally in your responses without directly quoting or referencing the source.
"""
        
        # Add session context if available
        if self.session_context:
            system_prompt += f"""
SESSION CONTEXT:
{json.dumps(self.session_context, indent=2)}
"""
        
        # Add conversation continuity if history exists
        if user_history and len(user_history) > 0:
            system_prompt += """
CONVERSATION CONTINUITY:
Remember previous topics discussed and build upon them naturally. Show that you remember the user's concerns and progress.
"""
        
        return system_prompt
    
    def generate_user_prompt(self, user_input: str, additional_context: Dict = None) -> str:
        """
        Format user input with any additional context
        
        Args:
            user_input: Raw user input
            additional_context: Any additional context (mood, time of day, etc.)
            
        Returns:
            Formatted user prompt
        """
        prompt = user_input
        
        if additional_context:
            if 'mood' in additional_context:
                prompt = f"[User mood: {additional_context['mood']}] {prompt}"
            if 'urgency' in additional_context:
                prompt = f"[Urgency: {additional_context['urgency']}] {prompt}"
                
        return prompt
    
    def create_conversation_messages(self, 
                                   user_input: str,
                                   therapy_type: Optional[TherapyType] = None,
                                   pdf_context: str = "",
                                   conversation_history: List[Dict] = None,
                                   additional_context: Dict = None) -> List[Dict]:
        """
        Create complete message array for OpenAI API
        
        Args:
            user_input: Current user input
            therapy_type: Therapy approach to use
            pdf_context: Relevant PDF context
            conversation_history: Previous messages
            additional_context: Additional context
            
        Returns:
            List of message dictionaries for OpenAI API
        """
        messages = []
        
        # Add system prompt
        system_prompt = self.generate_system_prompt(
            therapy_type=therapy_type,
            pdf_context=pdf_context,
            user_history=conversation_history
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if exists
        if conversation_history:
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                messages.append(msg)
        
        # Add current user input
        user_prompt = self.generate_user_prompt(user_input, additional_context)
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def get_crisis_response(self) -> str:
        """Get crisis intervention response"""
        return """I'm deeply concerned about what you're sharing. Your safety is the most important thing right now.

Please reach out to immediate support:
- **Crisis Hotline**: 988 (US) or your local emergency number
- **Emergency Services**: 911 or your local emergency number
- **Crisis Text Line**: Text HOME to 741741

You don't have to go through this alone. These services have trained professionals available 24/7 who can provide immediate support.

Would you like me to help you think through reaching out to one of these resources, or is there someone trusted you could contact right now?"""
    
    def detect_crisis_keywords(self, text: str) -> bool:
        """
        Detect potential crisis situations in user input
        
        Args:
            text: User input text
            
        Returns:
            Boolean indicating potential crisis
        """
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'not worth living',
            'harm myself', 'self harm', 'cutting', 'overdose',
            'plan to die', 'better off dead', 'no point in living'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    def get_therapy_suggestions(self, user_input: str, current_therapy: TherapyType) -> List[TherapyType]:
        """
        Suggest appropriate therapy types based on user input
        
        Args:
            user_input: User's message
            current_therapy: Currently selected therapy type
            
        Returns:
            List of suggested therapy types
        """
        suggestions = [current_therapy]
        input_lower = user_input.lower()
        
        # Keyword mapping to therapy types
        keyword_therapy_map = {
            'anxious|worry|panic|nervous': TherapyType.ANXIETY,
            'sad|depressed|hopeless|empty': TherapyType.DEPRESSION,
            'loss|died|death|grief|mourning': TherapyType.GRIEF,
            'trauma|ptsd|flashback|nightmare': TherapyType.TRAUMA,
            'child|kid|parenting|family': TherapyType.PARENTING,
            'thought|thinking|cognitive|belief': TherapyType.CBT,
            'emotion|feeling|mindful|regulate': TherapyType.DBT,
            'value|meaning|accept|commit': TherapyType.ACT
        }
        
        for keywords, therapy_type in keyword_therapy_map.items():
            if any(word in input_lower for word in keywords.split('|')):
                if therapy_type not in suggestions:
                    suggestions.append(therapy_type)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def format_therapeutic_response(self, ai_response: str, include_disclaimer: bool = True) -> str:
        """
        Format AI response with appropriate therapeutic framing
        
        Args:
            ai_response: Raw AI response
            include_disclaimer: Whether to include disclaimer
            
        Returns:
            Formatted response
        """
        formatted_response = ai_response
        
        if include_disclaimer and self.session_context.get('first_message', True):
            disclaimer = "\n\n*Note: I'm an AI therapy support assistant. While I aim to provide helpful support, I'm not a replacement for professional mental health care. If you're experiencing severe distress, please consider reaching out to a licensed therapist.*"
            formatted_response += disclaimer
            self.session_context['first_message'] = False
            
        return formatted_response


# Example usage
if __name__ == "__main__":
    # Initialize prompt manager
    prompt_manager = PromptManager(
        default_therapy_type=TherapyType.CBT,
        conversation_style=ConversationStyle.EMPATHETIC
    )
    
    # Set session context
    prompt_manager.set_session_context({
        'user_name': 'Anonymous',
        'session_number': 1,
        'primary_concern': 'anxiety'
    })
    
    # Generate prompts
    user_input = "I've been feeling really anxious about work lately"
    pdf_context = "Anxiety often manifests as worry about future events..."
    
    messages = prompt_manager.create_conversation_messages(
        user_input=user_input,
        therapy_type=TherapyType.ANXIETY,
        pdf_context=pdf_context
    )
    
    print("Generated messages for AI:")
    for msg in messages:
        print(f"\n{msg['role'].upper()}:")
        print(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])
import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import base64
from dataclasses import dataclass
from enum import Enum
import tempfile
import wave

# OpenAI
import openai
from openai import OpenAI

# ElevenLabs
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

# Audio processing
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import io

# Local imports
from pdf_processor import PDFVectorStore
from prompt import PromptManager, TherapyType, ConversationStyle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Audio format options"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"


@dataclass
class TherapyResponse:
    """Data class for therapy response"""
    text: str
    audio_data: Optional[bytes] = None
    audio_format: Optional[AudioFormat] = None
    therapy_type: Optional[TherapyType] = None
    confidence_score: float = 1.0
    processing_time: float = 0.0
    context_used: List[Dict] = None


@dataclass
class AudioInput:
    """Data class for audio input"""
    audio_data: bytes
    format: AudioFormat
    language: str = "en"
    duration: Optional[float] = None


class EmothriveAI:
    """
    Main AI engine that integrates PDF knowledge, OpenAI GPT-4, and ElevenLabs TTS/STT.
    Provides therapy-focused conversational AI with multilingual support.
    """
    
    def __init__(self,
                 openai_api_key: str,
                 elevenlabs_api_key: str,
                 pdf_folder: str = "./pdf/",
                 default_therapy_type: TherapyType = TherapyType.GENERAL,
                 voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                 model: str = "gpt-4o-mini",
                 enable_crisis_detection: bool = True):
        """
        Initialize EmothriveAI with all necessary components
        
        Args:
            openai_api_key: OpenAI API key
            elevenlabs_api_key: ElevenLabs API key
            pdf_folder: Path to therapy PDF documents
            default_therapy_type: Default therapy approach
            voice_id: ElevenLabs voice ID
            model: OpenAI model to use
            enable_crisis_detection: Whether to detect crisis situations
        """
        # API clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
        
        # Initialize components
        self.pdf_store = PDFVectorStore(folder_path=pdf_folder, openai_api_key=openai_api_key)
        self.prompt_manager = PromptManager(
            default_therapy_type=default_therapy_type,
            conversation_style=ConversationStyle.EMPATHETIC
        )
        
        # Configuration
        self.voice_id = voice_id
        self.model = model
        self.enable_crisis_detection = enable_crisis_detection
        
        # Conversation state
        self.conversation_history: List[Dict] = []
        self.session_data = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': datetime.now(),
            'messages_count': 0,
            'therapy_types_used': set()
        }
        
        # Initialize PDF vector store
        self._initialize_knowledge_base()
        
        logger.info(f"EmothriveAI initialized with model: {self.model}")
    
    def _initialize_knowledge_base(self):
        """Initialize or load the PDF vector store"""
        try:
            # Try to load existing vector store
            if not self.pdf_store.load_vector_store():
                logger.info("Building vector store from PDFs...")
                self.pdf_store.build_vector_store()
            
            stats = self.pdf_store.get_stats()
            logger.info(f"Knowledge base ready: {stats['total_pdfs']} PDFs, "
                       f"{stats.get('total_chunks', 0)} chunks indexed")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            logger.warning("Continuing without PDF knowledge base")
    
    async def handle_input(self, 
                          user_input: Union[str, AudioInput],
                          therapy_type: Optional[TherapyType] = None,
                          return_audio: bool = True,
                          audio_format: AudioFormat = AudioFormat.MP3,
                          additional_context: Dict = None) -> TherapyResponse:
        """
        Main method to handle user input (text or audio) and generate therapy response
        
        Args:
            user_input: Text string or AudioInput object
            therapy_type: Specific therapy type to use
            return_audio: Whether to generate audio response
            audio_format: Format for audio output
            additional_context: Additional context (mood, urgency, etc.)
            
        Returns:
            TherapyResponse object with text and optional audio
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Process input (convert audio to text if needed)
            if isinstance(user_input, AudioInput):
                text_input = await self.transcribe_audio(user_input)
                logger.info(f"Transcribed audio: {text_input[:50]}...")
            else:
                text_input = user_input
            
            # Step 2: Crisis detection
            if self.enable_crisis_detection and self.prompt_manager.detect_crisis_keywords(text_input):
                crisis_response = self.prompt_manager.get_crisis_response()
                audio_data = None
                if return_audio:
                    audio_data = await self.generate_speech(crisis_response, audio_format)
                
                return TherapyResponse(
                    text=crisis_response,
                    audio_data=audio_data,
                    audio_format=audio_format,
                    therapy_type=TherapyType.GENERAL,
                    confidence_score=1.0,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Step 3: Get relevant context from PDFs
            pdf_context = self.pdf_store.get_context_for_prompt(text_input, k=3)
            
            # Step 4: Determine therapy type if not specified
            if not therapy_type:
                suggestions = self.prompt_manager.get_therapy_suggestions(
                    text_input, 
                    self.prompt_manager.default_therapy_type
                )
                therapy_type = suggestions[0]
            
            # Step 5: Generate AI response
            messages = self.prompt_manager.create_conversation_messages(
                user_input=text_input,
                therapy_type=therapy_type,
                pdf_context=pdf_context,
                conversation_history=self.conversation_history,
                additional_context=additional_context
            )
            
            # Call OpenAI API
            response_text = await self._call_openai(messages)
            
            # Step 6: Format response
            formatted_response = self.prompt_manager.format_therapeutic_response(
                response_text,
                include_disclaimer=(self.session_data['messages_count'] == 0)
            )
            
            # Step 7: Generate audio if requested
            audio_data = None
            if return_audio:
                audio_data = await self.generate_speech(formatted_response, audio_format)
            
            # Step 8: Update conversation history
            self._update_conversation_history(text_input, formatted_response)
            
            # Step 9: Create response object
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TherapyResponse(
                text=formatted_response,
                audio_data=audio_data,
                audio_format=audio_format if audio_data else None,
                therapy_type=therapy_type,
                confidence_score=0.95,  # Could be calculated based on context relevance
                processing_time=processing_time,
                context_used=[{
                    'source': doc['source'],
                    'therapy_type': doc['therapy_type']
                } for doc in self.pdf_store.get_similar_docs(text_input, k=3)]
            )
            
        except Exception as e:
            logger.error(f"Error handling input: {e}")
            error_response = "I apologize, but I encountered an error processing your message. Please try again or rephrase your concern."
            
            return TherapyResponse(
                text=error_response,
                therapy_type=therapy_type or TherapyType.GENERAL,
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _call_openai(self, messages: List[Dict]) -> str:
        """
        Call OpenAI API with messages
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            AI response text
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def transcribe_audio(self, audio_input: AudioInput) -> str:
        """
        Transcribe audio to text using ElevenLabs Scribe
        
        Args:
            audio_input: AudioInput object with audio data
            
        Returns:
            Transcribed text
        """
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{audio_input.format.value}", delete=False) as tmp_file:
                tmp_file.write(audio_input.audio_data)
                tmp_path = tmp_file.name
            
            # Use ElevenLabs transcription (Scribe)
            # Note: As of my knowledge, ElevenLabs primarily offers TTS
            # For STT, we'll use OpenAI Whisper as fallback
            with open(tmp_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=audio_input.language
                )
            
            # Clean up
            os.unlink(tmp_path)
            
            return transcript.text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    async def generate_speech(self, 
                            text: str, 
                            audio_format: AudioFormat = AudioFormat.MP3,
                            voice_settings: Optional[VoiceSettings] = None) -> bytes:
        """
        Generate speech from text using ElevenLabs Multilingual v2
        
        Args:
            text: Text to convert to speech
            audio_format: Output audio format
            voice_settings: Optional voice settings
            
        Returns:
            Audio data as bytes
        """
        try:
            # Default voice settings for therapy context
            if not voice_settings:
                voice_settings = VoiceSettings(
                    stability=0.75,  # Balanced stability for natural sound
                    similarity_boost=0.75,  # Good similarity to selected voice
                    style=0.5,  # Moderate style for therapeutic tone
                    use_speaker_boost=True
                )
            
            # Generate audio using ElevenLabs
            audio_generator = self.elevenlabs_client.generate(
                text=text,
                voice=self.voice_id,
                voice_settings=voice_settings,
                model="eleven_multilingual_v2"  # Multilingual model
            )
            
            # Collect audio chunks
            audio_chunks = []
            for chunk in audio_generator:
                audio_chunks.append(chunk)
            
            # Combine chunks
            audio_data = b''.join(audio_chunks)
            
            # Convert format if needed
            if audio_format != AudioFormat.MP3:
                audio_data = self._convert_audio_format(audio_data, AudioFormat.MP3, audio_format)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            raise
    
    def _convert_audio_format(self, audio_data: bytes, from_format: AudioFormat, to_format: AudioFormat) -> bytes:
        """Convert audio between formats"""
        try:
            # Load audio
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=from_format.value)
            
            # Convert to target format
            output = io.BytesIO()
            audio.export(output, format=to_format.value)
            output.seek(0)
            
            return output.read()
            
        except Exception as e:
            logger.error(f"Audio format conversion error: {e}")
            return audio_data  # Return original if conversion fails
    
    def _update_conversation_history(self, user_input: str, ai_response: str):
        """Update conversation history and session data"""
        # Add to history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Keep only last 20 messages (10 exchanges) for context
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        # Update session data
        self.session_data['messages_count'] += 1
        
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        return {
            'session_id': self.session_data['session_id'],
            'duration': (datetime.now() - self.session_data['start_time']).total_seconds(),
            'messages_count': self.session_data['messages_count'],
            'therapy_types_used': list(self.session_data['therapy_types_used']),
            'knowledge_base_stats': self.pdf_store.get_stats()
        }
    
    def export_conversation(self, format: str = "json") -> Union[str, Dict]:
        """Export conversation history"""
        export_data = {
            'session': self.session_data,
            'conversation': self.conversation_history,
            'timestamp': datetime.now().isoformat()
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            return export_data
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        self.session_data = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': datetime.now(),
            'messages_count': 0,
            'therapy_types_used': set()
        }
        logger.info("Conversation reset")


# Backend integration class
class EmothriveBackendInterface:
    """
    Interface for backend integration
    Provides REST API-friendly methods
    """
    
    def __init__(self, ai_engine: EmothriveAI):
        self.ai_engine = ai_engine
    
    async def process_message(self, request_data: Dict) -> Dict:
        """
        Process message from backend
        
        Expected request_data format:
        {
            "message": "user text or base64 audio",
            "message_type": "text" or "audio",
            "therapy_type": "CBT", "DBT", etc. (optional),
            "return_audio": true/false,
            "audio_format": "mp3", "wav", etc.,
            "user_context": {
                "mood": "anxious",
                "urgency": "low"
            }
        }
        """
        try:
            # Parse input
            message = request_data.get('message')
            message_type = request_data.get('message_type', 'text')
            therapy_type_str = request_data.get('therapy_type')
            return_audio = request_data.get('return_audio', True)
            audio_format_str = request_data.get('audio_format', 'mp3')
            user_context = request_data.get('user_context', {})
            
            # Convert therapy type
            therapy_type = None
            if therapy_type_str:
                therapy_type = TherapyType[therapy_type_str.upper()]
            
            # Convert audio format
            audio_format = AudioFormat(audio_format_str.lower())
            
            # Process based on message type
            if message_type == 'audio':
                # Decode base64 audio
                audio_data = base64.b64decode(message)
                user_input = AudioInput(
                    audio_data=audio_data,
                    format=audio_format,
                    language=user_context.get('language', 'en')
                )
            else:
                user_input = message
            
            # Get response
            response = await self.ai_engine.handle_input(
                user_input=user_input,
                therapy_type=therapy_type,
                return_audio=return_audio,
                audio_format=audio_format,
                additional_context=user_context
            )
            
            # Format response for backend
            response_data = {
                'success': True,
                'response': {
                    'text': response.text,
                    'audio': base64.b64encode(response.audio_data).decode() if response.audio_data else None,
                    'audio_format': response.audio_format.value if response.audio_format else None,
                    'therapy_type': response.therapy_type.name if response.therapy_type else None,
                    'confidence': response.confidence_score,
                    'processing_time': response.processing_time,
                    'context_sources': response.context_used or []
                },
                'session': {
                    'session_id': self.ai_engine.session_data['session_id'],
                    'message_count': self.ai_engine.session_data['messages_count']
                }
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Backend interface error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': {
                    'text': "I apologize, but I encountered an error. Please try again.",
                    'audio': None
                }
            }


# Example usage for backend integration
async def main():
    """Example of how to use the AI engine"""
    # Initialize AI engine
    ai_engine = EmothriveAI(
        openai_api_key="your-openai-api-key",
        elevenlabs_api_key="your-elevenlabs-api-key",
        pdf_folder="./pdf/",
        default_therapy_type=TherapyType.CBT
    )
    
    # Create backend interface
    backend = EmothriveBackendInterface(ai_engine)
    
    # Example request from backend
    request = {
        "message": "I've been feeling really anxious about my job interview tomorrow",
        "message_type": "text",
        "therapy_type": "ANXIETY",
        "return_audio": True,
        "audio_format": "mp3",
        "user_context": {
            "mood": "anxious",
            "urgency": "medium"
        }
    }
    
    # Process request
    response = await backend.process_message(request)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
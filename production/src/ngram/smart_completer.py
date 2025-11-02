"""
Smart Code Completion Engine v2.0
Enhanced with better accuracy, logic, and comprehensive Python syntax
"""
import re
from typing import List, Tuple, Optional, Dict
from .hybrid_model import HybridCodeCompleter


class SmartCodeCompleter:
    """
    Enhanced code completer that understands user intent
    and generates complete code blocks with Python syntax knowledge
    """
    
    def __init__(self, hybrid_completer: HybridCodeCompleter):
        """Initialize with base hybrid completer"""
        self.completer = hybrid_completer
        self.templates = self._load_templates()
        self.syntax_patterns = self._load_syntax_patterns()
        self.naming_conventions = self._load_naming_conventions()
    
    def _load_templates(self) -> Dict[str, Dict]:
        """Load comprehensive code templates with Python syntax"""
        return {
            # Mathematical operations
            'add': {
                'patterns': [r'def\s+add', r'def\s+sum', r'def\s+plus'],
                'template': '''def add(a, b):
    """Add two numbers"""
    return a + b''',
                'description': 'Add two numbers'
            },
            'subtract': {
                'patterns': [r'def\s+sub', r'def\s+subtract', r'def\s+minus'],
                'template': '''def subtract(a, b):
    """Subtract b from a"""
    return a - b''',
                'description': 'Subtract two numbers'
            },
            'multiply': {
                'patterns': [r'def\s+mul', r'def\s+multiply', r'def\s+times'],
                'template': '''def multiply(a, b):
    """Multiply two numbers"""
    return a * b''',
                'description': 'Multiply two numbers'
            },
            'divide': {
                'patterns': [r'def\s+div', r'def\s+divide'],
                'template': '''def divide(a, b):
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b''',
                'description': 'Divide two numbers with zero check'
            },
            'power': {
                'patterns': [r'def\s+pow', r'def\s+power'],
                'template': '''def power(base, exponent):
    """Calculate power of a number"""
    return base ** exponent''',
                'description': 'Calculate power'
            },
            
            # List operations
            'sort': {
                'patterns': [r'def\s+sort', r'def\s+order'],
                'template': '''def sort_list(lst, reverse=False):
    """Sort a list"""
    return sorted(lst, reverse=reverse)''',
                'description': 'Sort a list'
            },
            'reverse': {
                'patterns': [r'def\s+reverse', r'def\s+flip'],
                'template': '''def reverse_list(lst):
    """Reverse a list"""
    return lst[::-1]''',
                'description': 'Reverse a list'
            },
            'filter': {
                'patterns': [r'def\s+filter'],
                'template': '''def filter_list(lst, condition):
    """Filter list by condition"""
    return [item for item in lst if condition(item)]''',
                'description': 'Filter list by condition'
            },
            'map': {
                'patterns': [r'def\s+map', r'def\s+transform'],
                'template': '''def map_list(lst, func):
    """Apply function to each item"""
    return [func(item) for item in lst]''',
                'description': 'Map function over list'
            },
            'max': {
                'patterns': [r'def\s+max', r'def\s+maximum'],
                'template': '''def find_max(lst):
    """Find maximum value in list"""
    if not lst:
        return None
    return max(lst)''',
                'description': 'Find maximum value'
            },
            'min': {
                'patterns': [r'def\s+min', r'def\s+minimum'],
                'template': '''def find_min(lst):
    """Find minimum value in list"""
    if not lst:
        return None
    return min(lst)''',
                'description': 'Find minimum value'
            },
            
            # String operations
            'uppercase': {
                'patterns': [r'def\s+upper', r'def\s+to_upper'],
                'template': '''def to_uppercase(text):
    """Convert text to uppercase"""
    return text.upper()''',
                'description': 'Convert to uppercase'
            },
            'lowercase': {
                'patterns': [r'def\s+lower', r'def\s+to_lower'],
                'template': '''def to_lowercase(text):
    """Convert text to lowercase"""
    return text.lower()''',
                'description': 'Convert to lowercase'
            },
            'capitalize': {
                'patterns': [r'def\s+capitalize', r'def\s+title'],
                'template': '''def capitalize_text(text):
    """Capitalize first letter of each word"""
    return text.title()''',
                'description': 'Capitalize words'
            },
            'split': {
                'patterns': [r'def\s+split'],
                'template': '''def split_text(text, delimiter=' '):
    """Split text by delimiter"""
    return text.split(delimiter)''',
                'description': 'Split text'
            },
            'join': {
                'patterns': [r'def\s+join'],
                'template': '''def join_list(lst, separator=' '):
    """Join list items into string"""
    return separator.join(str(item) for item in lst)''',
                'description': 'Join list items'
            },
            'strip': {
                'patterns': [r'def\s+strip', r'def\s+trim'],
                'template': '''def strip_whitespace(text):
    """Remove leading/trailing whitespace"""
    return text.strip()''',
                'description': 'Strip whitespace'
            },
            
            # Search/Find operations
            'find': {
                'patterns': [r'def\s+find', r'def\s+search'],
                'template': '''def find_item(lst, target):
    """Find item in list"""
    try:
        return lst.index(target)
    except ValueError:
        return -1''',
                'description': 'Find item in list'
            },
            'contains': {
                'patterns': [r'def\s+contains', r'def\s+has'],
                'template': '''def contains(lst, item):
    """Check if list contains item"""
    return item in lst''',
                'description': 'Check if item exists'
            },
            'count': {
                'patterns': [r'def\s+count'],
                'template': '''def count_occurrences(lst, item):
    """Count occurrences of item"""
    return lst.count(item)''',
                'description': 'Count occurrences'
            },
            
            # Control flow - if/elif/else
            'if_statement': {
                'patterns': [r'^if\s+', r'\nif\s+'],
                'template': '''if condition:
    # Code to execute if true
    pass''',
                'description': 'If statement'
            },
            'if_else': {
                'patterns': [r'if\s+.*:\s*\n.*\nelse'],
                'template': '''if condition:
    # True branch
    pass
else:
    # False branch
    pass''',
                'description': 'If-else statement'
            },
            'if_elif_else': {
                'patterns': [r'if\s+.*:\s*\n.*\nelif'],
                'template': '''if condition1:
    # First condition
    pass
elif condition2:
    # Second condition
    pass
else:
    # Default case
    pass''',
                'description': 'If-elif-else statement'
            },
            
            # Exception handling
            'try_except': {
                'patterns': [r'^try:', r'\ntry:'],
                'template': '''try:
    # Code that might raise exception
    pass
except Exception as e:
    # Handle exception
    print(f"Error: {e}")''',
                'description': 'Try-except block'
            },
            'try_except_finally': {
                'patterns': [r'try:.*except.*finally'],
                'template': '''try:
    # Code that might raise exception
    pass
except Exception as e:
    # Handle exception
    print(f"Error: {e}")
finally:
    # Always executed
    pass''',
                'description': 'Try-except-finally block'
            },
            'try_except_else': {
                'patterns': [r'try:.*except.*else'],
                'template': '''try:
    # Code that might raise exception
    pass
except Exception as e:
    # Handle exception
    print(f"Error: {e}")
else:
    # Executed if no exception
    pass''',
                'description': 'Try-except-else block'
            },
            
            # Context managers
            'with_statement': {
                'patterns': [r'^with\s+', r'\nwith\s+'],
                'template': '''with open(filename, 'r') as f:
    # File operations
    content = f.read()''',
                'description': 'With statement (context manager)'
            },
            
            # Class templates
            'class': {
                'patterns': [r'class\s+\w+'],
                'template': '''class MyClass:
    """Class description"""
    
    def __init__(self):
        """Initialize the class"""
        pass
    
    def method(self):
        """Method description"""
        pass''',
                'description': 'Basic class structure'
            },
            'dataclass': {
                'patterns': [r'@dataclass', r'from\s+dataclasses\s+import\s+dataclass'],
                'template': '''from dataclasses import dataclass

@dataclass
class Person:
    """Data class for person"""
    name: str
    age: int
    email: str = ""''',
                'description': 'Data class'
            },
            
            # Loop templates
            'for_range': {
                'patterns': [r'for\s+\w+\s+in\s+range'],
                'template': '''for i in range(n):
    # Loop body
    pass''',
                'description': 'For loop with range'
            },
            'for_list': {
                'patterns': [r'for\s+\w+\s+in\s+\w+:'],
                'template': '''for item in items:
    # Process item
    print(item)''',
                'description': 'For loop over list'
            },
            'for_enumerate': {
                'patterns': [r'for\s+.*\s+in\s+enumerate'],
                'template': '''for index, item in enumerate(items):
    # Access both index and item
    print(f"{index}: {item}")''',
                'description': 'For loop with enumerate'
            },
            'while_loop': {
                'patterns': [r'while\s+'],
                'template': '''while condition:
    # Loop body
    pass''',
                'description': 'While loop'
            },
            
            # Comprehensions
            'list_comp': {
                'patterns': [r'\[\w+\s+for\s+', r'list.*comprehension'],
                'template': '''result = [item for item in items if condition]''',
                'description': 'List comprehension'
            },
            'dict_comp': {
                'patterns': [r'\{\w+:\s*\w+\s+for\s+', r'dict.*comprehension'],
                'template': '''result = {key: value for key, value in items.items() if condition}''',
                'description': 'Dictionary comprehension'
            },
            'set_comp': {
                'patterns': [r'\{\w+\s+for\s+', r'set.*comprehension'],
                'template': '''result = {item for item in items if condition}''',
                'description': 'Set comprehension'
            },
            
            # Decorators
            'decorator': {
                'patterns': [r'@\w+', r'def\s+decorator'],
                'template': '''def decorator(func):
    """Decorator function"""
    def wrapper(*args, **kwargs):
        # Pre-processing
        result = func(*args, **kwargs)
        # Post-processing
        return result
    return wrapper''',
                'description': 'Function decorator'
            },
            'property_decorator': {
                'patterns': [r'@property'],
                'template': '''@property
def value(self):
    """Get value"""
    return self._value

@value.setter
def value(self, new_value):
    """Set value"""
    self._value = new_value''',
                'description': 'Property decorator'
            },
            
            # Lambda functions
            'lambda': {
                'patterns': [r'lambda\s+'],
                'template': '''lambda x: x * 2''',
                'description': 'Lambda function'
            },
            
            # Generators
            'generator': {
                'patterns': [r'def\s+\w+.*yield'],
                'template': '''def generator():
    """Generator function"""
    for i in range(10):
        yield i''',
                'description': 'Generator function'
            },
            
            # File operations
            'read_file': {
                'patterns': [r'def\s+read', r'def\s+load'],
                'template': '''def read_file(filepath):
    """Read file contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()''',
                'description': 'Read file contents'
            },
            'write_file': {
                'patterns': [r'def\s+write', r'def\s+save'],
                'template': '''def write_file(filepath, content):
    """Write content to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)''',
                'description': 'Write to file'
            },
            'read_json': {
                'patterns': [r'def\s+read_json', r'def\s+load_json'],
                'template': '''def read_json(filepath):
    """Read JSON file"""
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)''',
                'description': 'Read JSON file'
            },
            'write_json': {
                'patterns': [r'def\s+write_json', r'def\s+save_json'],
                'template': '''def write_json(filepath, data):
    """Write data to JSON file"""
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)''',
                'description': 'Write JSON file'
            },
        }
    
    def _load_syntax_patterns(self) -> Dict[str, str]:
        """Load Python syntax completion patterns"""
        return {
            'if': 'if condition:\n    pass',
            'else': 'else:\n    pass',
            'elif': 'elif condition:\n    pass',
            'for': 'for item in items:\n    pass',
            'while': 'while condition:\n    pass',
            'def': 'def function_name():\n    pass',
            'class': 'class ClassName:\n    pass',
            'try': 'try:\n    pass\nexcept Exception as e:\n    pass',
            'with': 'with open(file) as f:\n    pass',
            'return': 'return value',
            'yield': 'yield value',
            'raise': 'raise Exception("message")',
            'import': 'import module',
            'from': 'from module import function',
        }
    
    def _load_naming_conventions(self) -> Dict[str, Dict]:
        """Load naming convention patterns for better intent detection"""
        return {
            'get_': {
                'pattern': r'def\s+get_(\w+)',
                'template': '''def get_{name}(self):
    """Get {name}"""
    return self._{name}''',
                'description': 'Getter method'
            },
            'set_': {
                'pattern': r'def\s+set_(\w+)',
                'template': '''def set_{name}(self, value):
    """Set {name}"""
    self._{name} = value''',
                'description': 'Setter method'
            },
            'is_': {
                'pattern': r'def\s+is_(\w+)',
                'template': '''def is_{name}(self):
    """Check if {name}"""
    return self._{name}''',
                'description': 'Boolean checker method'
            },
            'has_': {
                'pattern': r'def\s+has_(\w+)',
                'template': '''def has_{name}(self):
    """Check if has {name}"""
    return self._{name} is not None''',
                'description': 'Has checker method'
            },
            'validate_': {
                'pattern': r'def\s+validate_(\w+)',
                'template': '''def validate_{name}(self, value):
    """Validate {name}"""
    if not value:
        raise ValueError("{name} cannot be empty")
    return True''',
                'description': 'Validation method'
            },
            'calculate_': {
                'pattern': r'def\s+calculate_(\w+)',
                'template': '''def calculate_{name}(self):
    """Calculate {name}"""
    return self._value * 2''',
                'description': 'Calculation method'
            },
        }
    
    def detect_intent(self, code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect user intent from partial code with enhanced logic
        
        Returns tuple of (template_name, extracted_name) if matched, (None, None) otherwise
        """
        code_lower = code.lower().strip()
        
        # First check naming conventions (get_, set_, is_, has_, etc.)
        for conv_prefix, conv_data in self.naming_conventions.items():
            match = re.search(conv_data['pattern'], code_lower)
            if match:
                extracted_name = match.group(1) if match.lastindex else None
                return (conv_prefix.rstrip('_'), extracted_name)
        
        # Then check templates
        for template_name, template_data in self.templates.items():
            for pattern in template_data['patterns']:
                if re.search(pattern, code_lower):
                    return (template_name, None)
        
        return (None, None)
    
    def get_smart_completion(self, code: str, max_lines: int = 10) -> Dict:
        """
        Get smart code completion with enhanced accuracy and logic
        
        Returns dict with:
        - suggestions: List of token suggestions
        - full_completion: Complete code block if detected
        - intent: Detected intent
        - confidence: Confidence score (0.0-1.0)
        - description: Human-readable description
        """
        result = {
            'suggestions': [],
            'full_completion': None,
            'intent': None,
            'confidence': 0.0,
            'description': None
        }
        
        # Get basic token suggestions
        suggestions = self.completer.suggest(code, top_k=5, use_lstm=True)
        result['suggestions'] = suggestions
        
        # Detect intent with enhanced logic
        intent, extracted_name = self.detect_intent(code)
        
        if intent:
            result['intent'] = intent
            
            # Check if it's a naming convention pattern
            if extracted_name and intent in ['get', 'set', 'is', 'has', 'validate', 'calculate']:
                # Use naming convention template
                conv_key = f'{intent}_'
                if conv_key in self.naming_conventions:
                    conv_data = self.naming_conventions[conv_key]
                    template = conv_data['template'].replace('{name}', extracted_name)
                    result['description'] = conv_data['description'].replace('{name}', extracted_name)
                    result['full_completion'] = template
                    result['confidence'] = 0.92  # High confidence for naming conventions
            # Use regular template
            elif intent in self.templates:
                template_data = self.templates[intent]
                result['description'] = template_data['description']
                
                # Generate full completion
                result['full_completion'] = self._generate_full_completion(
                    code, template_data, suggestions
                )
                
                # Calculate confidence based on match quality
                result['confidence'] = self._calculate_confidence(code, template_data)
        
        # If no template match, try intelligent multi-line completion
        if not result['full_completion']:
            result['full_completion'] = self._generate_contextual_completion(
                code, suggestions, max_lines
            )
        
        return result
    
    def _generate_full_completion(
        self, 
        code: str, 
        template_data: Dict,
        suggestions: List[Tuple[str, float]]
    ) -> str:
        """Generate full code completion from template"""
        template = template_data['template']
        
        # Try to extract function name from partial code
        func_match = re.search(r'def\s+(\w+)', code)
        func_name = None
        if func_match:
            func_name = func_match.group(1)
            # Replace template function name with user's name
            template = re.sub(r'def\s+\w+', f'def {func_name}', template)
        
        # Try to extract class name
        class_match = re.search(r'class\s+(\w+)', code)
        if class_match:
            class_name = class_match.group(1)
            template = re.sub(r'class\s+\w+', f'class {class_name}', template)
        
        # If user has started parameters, preserve them
        param_match = re.search(r'def\s+\w+\s*\((.*)', code)
        if param_match and param_match.group(1).strip() and func_name:
            partial_params = param_match.group(1)
            # Try to complete parameters intelligently
            if suggestions and suggestions[0][0] == ')':
                # User likely wants to close params
                completed_params = partial_params + ')'
            else:
                completed_params = partial_params
            
            # Replace template params with user's partial params
            template = re.sub(
                r'def\s+\w+\s*\([^)]*\)',
                f'def {func_name}({completed_params})',
                template
            )
        
        return template
    
    def _generate_contextual_completion(
        self,
        code: str,
        suggestions: List[Tuple[str, float]],
        max_lines: int
    ) -> Optional[str]:
        """
        Generate multi-line completion based on context
        Without explicit template match
        """
        # Check if we're in a function definition
        if re.search(r'def\s+\w+\s*\([^)]*\)\s*:\s*$', code):
            # Function body needed
            return self._complete_function_body(code, suggestions)
        
        # Check if we're in a class definition
        if re.search(r'class\s+\w+.*:\s*$', code):
            return self._complete_class_body(code)
        
        # Check if we're in a loop
        if re.search(r'(for|while)\s+.*:\s*$', code):
            return self._complete_loop_body(code, suggestions)
        
        # Check if we're in an if statement
        if re.search(r'if\s+.*:\s*$', code):
            return self._complete_if_body(code, suggestions)
        
        # Default: use token-by-token completion
        return self._complete_token_by_token(code, suggestions, max_lines)
    
    def _complete_function_body(self, code: str, suggestions: List) -> str:
        """Complete function body intelligently"""
        # Extract function name to guess purpose
        func_match = re.search(r'def\s+(\w+)', code)
        if not func_match:
            return code + '\n    pass'
        
        func_name = func_match.group(1).lower()
        
        # Guess based on function name
        if 'get' in func_name or 'fetch' in func_name:
            body = '\n    """Get/fetch data"""\n    return None'
        elif 'set' in func_name or 'update' in func_name:
            body = '\n    """Set/update value"""\n    pass'
        elif 'calculate' in func_name or 'compute' in func_name:
            body = '\n    """Calculate result"""\n    result = 0\n    return result'
        elif 'print' in func_name or 'display' in func_name or 'show' in func_name:
            body = '\n    """Display information"""\n    print()'
        elif 'check' in func_name or 'validate' in func_name or 'is_' in func_name:
            body = '\n    """Check/validate condition"""\n    return True'
        else:
            body = '\n    """Function description"""\n    pass'
        
        return code + body
    
    def _complete_class_body(self, code: str) -> str:
        """Complete class body"""
        return code + '\n    """Class description"""\n    \n    def __init__(self):\n        """Initialize class"""\n        pass'
    
    def _complete_loop_body(self, code: str, suggestions: List) -> str:
        """Complete loop body"""
        return code + '\n    # Loop body\n    pass'
    
    def _complete_if_body(self, code: str, suggestions: List) -> str:
        """Complete if statement body"""
        return code + '\n    # Condition body\n    pass'
    
    def _complete_token_by_token(
        self,
        code: str,
        suggestions: List,
        max_tokens: int
    ) -> str:
        """Complete code token by token using suggestions"""
        current = code
        tokens_added = 0
        
        while tokens_added < max_tokens:
            sug = self.completer.suggest(current, top_k=1, use_lstm=True)
            if not sug or sug[0][1] < 0.1:  # Low confidence
                break
            
            token = sug[0][0]
            current += ' ' + token if token not in [')', ']', '}', ':', ',', '.'] else token
            tokens_added += 1
            
            # Stop at natural breakpoints
            if token in [':', '\n'] or (tokens_added > 3 and sug[0][1] < 0.3):
                break
        
        return current
    
    def _calculate_confidence(self, code: str, template_data: Dict) -> float:
        """Calculate confidence score for template match"""
        # Check pattern match strength
        max_confidence = 0.0
        
        for pattern in template_data['patterns']:
            match = re.search(pattern, code.lower())
            if match:
                # Full match = high confidence
                if match.group(0) == code.lower().strip():
                    max_confidence = max(max_confidence, 0.95)
                else:
                    # Partial match = medium confidence
                    match_ratio = len(match.group(0)) / len(code.strip())
                    max_confidence = max(max_confidence, 0.6 + 0.3 * match_ratio)
        
        return max_confidence
    
    def get_multiple_smart_completions(
        self,
        code: str,
        num_completions: int = 3
    ) -> List[Dict]:
        """Get multiple smart completion options"""
        results = []
        
        # Get main completion
        main_result = self.get_smart_completion(code)
        results.append(main_result)
        
        # If template matched, also provide token-by-token alternative
        if main_result['intent']:
            # Alternative 1: Just next few tokens
            alt1 = {
                'suggestions': main_result['suggestions'],
                'full_completion': self._complete_token_by_token(
                    code, main_result['suggestions'], 5
                ),
                'intent': None,
                'confidence': 0.5,
                'description': 'Token-by-token completion'
            }
            results.append(alt1)
            
            # Alternative 2: Minimal completion
            alt2 = {
                'suggestions': main_result['suggestions'],
                'full_completion': code + '\n    pass',
                'intent': None,
                'confidence': 0.3,
                'description': 'Minimal completion'
            }
            results.append(alt2)
        
        return results[:num_completions]

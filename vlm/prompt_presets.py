"""Editable VLM instructions and persistent named presets for the optional viewer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

from vlm.prompts import normalize_prompts
from vlm.propose import INSTRUCTION, instruction_for
from vlm.prompts import validate_target

DEFAULT_NAME = 'Built-in default'
DEFAULT_INSTRUCTION = INSTRUCTION.replace('{robot}', 'a small wheeled ground robot')


def validate_instruction(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('The VLM prompt cannot be empty.')
    return instruction


def instruction_settings(args):
    custom = getattr(args, 'vlm_instruction', None)
    target = validate_target(getattr(args, 'vlm_target', 'non_traversable'))
    instruction = validate_instruction(instruction_for(target) if custom is None else custom)
    name = getattr(args, 'vlm_prompt_name', None) or (DEFAULT_NAME if custom is None else 'Unsaved draft')
    return {'instruction': instruction, 'vlm_prompt_name': name, 'vlm_target': target,
            'vlm_instruction_sha256': hashlib.sha256(instruction.encode('utf-8')).hexdigest()}


def parse_sam_prompts(text, limit=8):
    prompts = normalize_prompts([part for part in re.split(r'[,\n\r]+', text) if part.strip()])
    return validate_sam_prompts(prompts, limit)


def validate_sam_prompts(prompts, limit=8):
    prompts = normalize_prompts(prompts)
    if not prompts:
        raise ValueError('Enter at least one SAM object name.')
    if len(prompts) > limit:
        raise ValueError(f'Use at most {limit} distinct SAM object names.')
    return prompts


class PromptPresets:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name):
        # Names are labels, never paths. This also supports punctuation and Unicode.
        return self.directory / (hashlib.sha256(name.encode('utf-8')).hexdigest()+'.json')

    def list(self):
        presets = {}
        for path in sorted(self.directory.glob('*.json')):
            value = json.loads(path.read_text(encoding='utf-8'))
            name = value['name']
            if self._path(name) != path or name == DEFAULT_NAME:
                raise ValueError(f'Invalid prompt preset: {path.name}')
            validate_instruction(value['instruction'])
            value['target'] = validate_target(value.get('target', 'non_traversable'))
            presets[name] = value
        return dict(sorted(presets.items()))

    def save(self, name, instruction, target='non_traversable'):
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError('Use a preset name of 1–100 characters.')
        if name == DEFAULT_NAME:
            raise ValueError('Choose a new name to save a variation of the built-in default.')
        instruction = validate_instruction(instruction)
        value = {'name': name, 'instruction': instruction, 'target': validate_target(target),
                 'instruction_sha256': hashlib.sha256(instruction.encode('utf-8')).hexdigest(),
                 'updated_at': datetime.now(timezone.utc).isoformat()}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.directory,
                                             prefix='.preset-', suffix='.tmp', delete=False) as file:
                temporary = Path(file.name)
                file.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
            temporary.replace(self._path(name))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return value

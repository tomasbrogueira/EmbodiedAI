"""Timestamped VLM concept lifetime; independent SAM queries live in run configuration."""
class PromptState:
    """Keep recent concepts, never carry a pixel mask between camera frames."""
    def __init__(self, ttl=5., max_prompts=8):
        self.ttl, self.max_prompts = ttl, max_prompts
        self.seen = {}

    def observe(self, prompts, source_time):
        for prompt in prompts:
            # These equivalent descriptions generated duplicate SAM queries
            # in the development videos. Keep one object concept online.
            prompt = {'fallen log': 'log', 'detached log': 'log'}.get(prompt, prompt)
            self.seen[prompt] = source_time

    def active(self, source_time):
        self.seen = {p: t for p, t in self.seen.items() if source_time - t <= self.ttl}
        return [p for p, _ in sorted(self.seen.items(), key=lambda pair: pair[1], reverse=True)][:self.max_prompts]



"""
Strategy Registry

Central registry for all trading strategies.
Use this to easily switch between strategies.

Usage:
    from src.strategies.strategy_registry import StrategyRegistry
    
    # Get the winning strategy
    engine = StrategyRegistry.get('v5_relaxed_scanner')
    
    # Or use the default (recommended)
    engine = StrategyRegistry.get_default()
"""

from .v5_strict import TickBacktestEngineV5


class StrategyRegistry:
    """Registry of all available strategies."""
    
    _strategies = {
        'v5_strict': {
            'class': TickBacktestEngineV5,
            'description': 'Original V5 strict entry (2-of-3 criteria)',
            'win_rate': '80.0%',
            'status': 'stable'
        },
        'v5_relaxed_scanner': {
            'class': TickBacktestEngineV5,  # Uses V5 entry
            'description': 'Relaxed scanner (30% gain) + V5 strict entry - RECOMMENDED',
            'win_rate': '78.9%',
            'status': 'recommended'
        },
    }
    
    @classmethod
    def get(cls, name: str):
        """
        Get a strategy by name.
        
        Args:
            name: Strategy name ('v5_strict', 'v5_relaxed_scanner', etc.)
            
        Returns:
            Strategy class instance
        """
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        
        strategy_class = cls._strategies[name]['class']
        return strategy_class()
    
    @classmethod
    def get_default(cls):
        """Get the recommended default strategy."""
        return cls.get('v5_relaxed_scanner')
    
    @classmethod
    def list_all(cls):
        """List all available strategies."""
        print("="*60)
        print("Available Strategies")
        print("="*60)
        for name, info in cls._strategies.items():
            status = f"[{info['status'].upper()}]"
            print(f"\n{name}")
            print(f"  Status: {status}")
            print(f"  Win Rate: {info['win_rate']}")
            print(f"  Description: {info['description']}")
    
    @classmethod
    def get_recommended(cls):
        """Get the recommended strategy for live trading."""
        for name, info in cls._strategies.items():
            if info['status'] == 'recommended':
                return cls.get(name)
        return cls.get_default()


# Convenience function for quick access
def get_strategy(name: str = 'v5_relaxed_scanner'):
    """
    Quick function to get a strategy.
    
    Args:
        name: Strategy name (default: 'v5_relaxed_scanner')
        
    Returns:
        Strategy instance
    """
    return StrategyRegistry.get(name)


if __name__ == "__main__":
    # Show all strategies when run directly
    StrategyRegistry.list_all()

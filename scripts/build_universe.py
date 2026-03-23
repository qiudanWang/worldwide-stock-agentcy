"""Build the full CN + US tech stock universe."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.universe.merge_universe import build_full_universe

if __name__ == "__main__":
    universe = build_full_universe()
    print(f"\nUniverse built: {len(universe)} total stocks")
    print(f"  CN: {len(universe[universe['market'] == 'CN'])}")
    print(f"  US: {len(universe[universe['market'] == 'US'])}")

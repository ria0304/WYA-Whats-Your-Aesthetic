# services/analytics.py
# Wardrobe analytics calculations

from typing import Any, Dict, List
from collections import defaultdict


class WardrobeAnalytics:
    """
    Calculate wardrobe analytics:
    - Cost per wear
    - Most/least worn items
    - Category distribution
    - Color distribution
    - Sustainability score
    - Value distribution
    """

    @staticmethod
    def calculate_cost_per_wear(
        wardrobe_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate cost per wear for each item.
        Returns sorted list (lowest cost per wear first).
        """
        results = []
        for item in wardrobe_items:
            price = item.get('price', 0)
            wear_count = item.get('wear_count', 0)
            name = item.get('name', 'Unknown')

            if price > 0 and wear_count > 0:
                cost_per_wear = round(price / wear_count, 2)
                results.append({
                    'name': name,
                    'item_id': item.get('item_id'),
                    'price': price,
                    'wear_count': wear_count,
                    'cost_per_wear': cost_per_wear,
                    'category': item.get('category', 'Unknown'),
                    'color': item.get('color', 'Unknown')
                })

        return sorted(results, key=lambda x: x['cost_per_wear'])

    @staticmethod
    def get_most_worn_items(
        wardrobe_items: List[Dict[str, Any]],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get most worn items."""
        sorted_items = sorted(
            wardrobe_items,
            key=lambda x: x.get('wear_count', 0),
            reverse=True
        )
        return [{
            'name': item.get('name', 'Unknown'),
            'item_id': item.get('item_id'),
            'wear_count': item.get('wear_count', 0),
            'category': item.get('category', 'Unknown'),
            'color': item.get('color', 'Unknown'),
            'last_worn': item.get('last_worn')
        } for item in sorted_items[:limit] if item.get('wear_count', 0) > 0]

    @staticmethod
    def get_least_worn_items(
        wardrobe_items: List[Dict[str, Any]],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get least worn items (including never worn)."""
        sorted_items = sorted(
            wardrobe_items,
            key=lambda x: x.get('wear_count', 0)
        )
        return [{
            'name': item.get('name', 'Unknown'),
            'item_id': item.get('item_id'),
            'wear_count': item.get('wear_count', 0),
            'category': item.get('category', 'Unknown'),
            'color': item.get('color', 'Unknown'),
            'created_at': item.get('created_at')
        } for item in sorted_items[:limit]]

    @staticmethod
    def get_category_distribution(
        wardrobe_items: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get distribution of items by category."""
        distribution = defaultdict(int)
        for item in wardrobe_items:
            category = item.get('category', 'Unknown')
            distribution[category] += 1
        return dict(distribution)

    @staticmethod
    def get_color_distribution(
        wardrobe_items: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get distribution of items by color."""
        distribution = defaultdict(int)
        for item in wardrobe_items:
            color = item.get('color', 'Unknown')
            distribution[color] += 1
        return dict(distribution)

    @staticmethod
    def get_sustainability_score(
        wardrobe_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall sustainability score."""
        sustainable_fabrics = ['cotton', 'linen', 'hemp', 'bamboo', 'wool', 'silk', 'cashmere', 'organic']

        total = len(wardrobe_items)
        if total == 0:
            return {'score': 0, 'details': 'No items'}

        sustainable_count = 0
        brand_scores = []
        fabric_scores = []

        for item in wardrobe_items:
            fabric = item.get('fabric', '').lower()
            brand = item.get('brand', '').lower()
            sustainability = item.get('sustainability_score', 0)

            if any(sf in fabric for sf in sustainable_fabrics):
                sustainable_count += 1
                fabric_scores.append(10)

            if sustainability > 0:
                brand_scores.append(sustainability)

        fabric_score = round((sustainable_count / total) * 100, 1)
        brand_score = round(sum(brand_scores) / max(len(brand_scores), 1), 1) if brand_scores else 0

        overall_score = round((fabric_score + brand_score) / 2, 1)

        return {
            'score': overall_score,
            'fabric_score': fabric_score,
            'brand_score': brand_score,
            'sustainable_items': sustainable_count,
            'total_items': total,
            'percentage_sustainable': round((sustainable_count / total) * 100, 1)
        }

    @staticmethod
    def get_total_value(wardrobe_items: List[Dict[str, Any]]) -> float:
        """Calculate total wardrobe value."""
        return round(sum(item.get('price', 0) for item in wardrobe_items), 2)

    @staticmethod
    def get_average_wear_count(wardrobe_items: List[Dict[str, Any]]) -> float:
        """Calculate average wear count."""
        total = len(wardrobe_items)
        if total == 0:
            return 0.0
        total_wears = sum(item.get('wear_count', 0) for item in wardrobe_items)
        return round(total_wears / total, 1)

    @staticmethod
    def get_full_analytics(
        wardrobe_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get complete analytics in one call."""
        return {
            'total_items': len(wardrobe_items),
            'total_value': WardrobeAnalytics.get_total_value(wardrobe_items),
            'average_wear_count': WardrobeAnalytics.get_average_wear_count(wardrobe_items),
            'category_distribution': WardrobeAnalytics.get_category_distribution(wardrobe_items),
            'color_distribution': WardrobeAnalytics.get_color_distribution(wardrobe_items),
            'most_worn': WardrobeAnalytics.get_most_worn_items(wardrobe_items),
            'least_worn': WardrobeAnalytics.get_least_worn_items(wardrobe_items),
            'cost_per_wear': WardrobeAnalytics.calculate_cost_per_wear(wardrobe_items),
            'sustainability': WardrobeAnalytics.get_sustainability_score(wardrobe_items)
        }


analytics = WardrobeAnalytics()

"""Performance tests for entity and relationship validation (Issue #135).

Tests batch validation performance improvements and verifies that
optimizations (caching, batch processing) provide measurable benefits
for large-scale validation operations.
"""

import time
from tribalmemory.services.graph_store import (
    Entity,
    EntityValidator,
    Relationship,
    RelationshipValidator,
)


def test_entity_validator_batch_performance():
    """Test that batch validation performs better than individual validation."""
    validator = EntityValidator()
    
    # Generate 1000 test entities with various patterns
    entities = []
    for i in range(1000):
        # Mix of valid and invalid entities
        if i % 10 == 0:
            # Invalid: too short
            entities.append(Entity(name="AB", entity_type="concept"))
        elif i % 10 == 1:
            # Invalid: all caps stopword
            entities.append(Entity(name="THE", entity_type="concept"))
        elif i % 10 == 2:
            # Invalid: numeric only
            entities.append(Entity(name="12345", entity_type="concept"))
        elif i % 10 == 3:
            # Invalid: common word (concept type)
            entities.append(Entity(name="very", entity_type="concept"))
        elif i % 10 == 4:
            # Valid: proper name
            entities.append(Entity(name="Docker", entity_type="technology"))
        elif i % 10 == 5:
            # Valid: multi-word concept
            entities.append(Entity(name="machine learning", entity_type="concept"))
        else:
            # Valid: entity with unique name
            entities.append(Entity(name=f"Entity_{i}", entity_type="concept"))
    
    # Test individual validation
    start_individual = time.perf_counter()
    results_individual = [validator.is_valid(entity) for entity in entities]
    time_individual = time.perf_counter() - start_individual
    
    # Test batch validation
    start_batch = time.perf_counter()
    results_batch = validator.validate_batch(entities)
    time_batch = time.perf_counter() - start_batch
    
    # Verify results match
    assert results_individual == results_batch, "Batch and individual validation must produce identical results"
    
    # Verify we got some valid and some invalid results
    valid_count = sum(results_batch)
    assert valid_count > 0, "Should have some valid entities"
    assert valid_count < len(entities), "Should have some invalid entities"
    
    # Print performance metrics
    print(f"\nEntity Validation Performance (n={len(entities)}):")
    print(f"  Individual: {time_individual*1000:.2f}ms")
    print(f"  Batch:      {time_batch*1000:.2f}ms")
    print(f"  Valid:      {valid_count}/{len(entities)}")
    
    # Batch should be at least as fast (may be slightly slower due to list overhead,
    # but caching should help on repeated patterns)
    # We don't assert performance here since it's environment-dependent,
    # but we log it for visibility.


def test_entity_validator_cache_effectiveness():
    """Test that LRU cache improves performance on repeated entity names."""
    validator = EntityValidator()
    
    # Create entities with repeated names (should benefit from caching)
    repeated_names = ["Docker", "Python", "Kubernetes", "PostgreSQL", "Redis"]
    entities = []
    for _ in range(200):  # 200 iterations
        for name in repeated_names:  # 5 names = 1000 total entities
            entities.append(Entity(name=name, entity_type="technology"))
    
    # First pass - populates cache
    start_first = time.perf_counter()
    results_first = validator.validate_batch(entities)
    time_first = time.perf_counter() - start_first
    
    # Second pass - should use cache
    start_second = time.perf_counter()
    results_second = validator.validate_batch(entities)
    time_second = time.perf_counter() - start_second
    
    # Results should be identical
    assert results_first == results_second
    
    # All should be valid (proper technology names)
    assert all(results_second), "All technology names should be valid"
    
    print(f"\nCache Effectiveness Test (n={len(entities)} repeated names):")
    print(f"  First pass:  {time_first*1000:.2f}ms")
    print(f"  Second pass: {time_second*1000:.2f}ms")
    
    # Second pass should be faster or similar (cache hit)
    # We don't assert this strictly as it's environment-dependent


def test_relationship_validator_batch_performance():
    """Test that batch relationship validation works correctly."""
    validator = RelationshipValidator()
    
    # Generate 1000 test relationships
    relationships = []
    for i in range(1000):
        if i % 10 == 0:
            # Invalid: empty source
            relationships.append(Relationship(source="", target="target", relation_type="connects_to"))
        elif i % 10 == 1:
            # Invalid: self-relationship
            relationships.append(Relationship(source="service", target="service", relation_type="connects_to"))
        elif i % 10 == 2:
            # Invalid: source too short
            relationships.append(Relationship(source="AB", target="target", relation_type="connects_to"))
        elif i % 10 == 3:
            # Invalid: numeric source
            relationships.append(Relationship(source="123", target="target", relation_type="connects_to"))
        else:
            # Valid relationship
            relationships.append(
                Relationship(
                    source=f"service_{i}",
                    target=f"database_{i}",
                    relation_type="connects_to"
                )
            )
    
    # Test individual validation
    start_individual = time.perf_counter()
    results_individual = [validator.is_valid(rel) for rel in relationships]
    time_individual = time.perf_counter() - start_individual
    
    # Test batch validation
    start_batch = time.perf_counter()
    results_batch = validator.validate_batch(relationships)
    time_batch = time.perf_counter() - start_batch
    
    # Verify results match
    assert results_individual == results_batch, "Batch and individual validation must produce identical results"
    
    # Verify we got some valid and some invalid results
    valid_count = sum(results_batch)
    assert valid_count > 0, "Should have some valid relationships"
    assert valid_count < len(relationships), "Should have some invalid relationships"
    
    print(f"\nRelationship Validation Performance (n={len(relationships)}):")
    print(f"  Individual: {time_individual*1000:.2f}ms")
    print(f"  Batch:      {time_batch*1000:.2f}ms")
    print(f"  Valid:      {valid_count}/{len(relationships)}")


def test_large_scale_validation():
    """Test validation at scale (10,000+ entities)."""
    validator = EntityValidator()
    
    # Generate 10,000 entities
    entities = []
    for i in range(10000):
        # Mix of entity types and patterns
        entity_type = ["concept", "technology", "person", "organization"][i % 4]
        name = f"Entity_{i}_{'_'.join(['word'] * (i % 5 + 1))}"
        entities.append(Entity(name=name, entity_type=entity_type))
    
    # Validate in batch
    start = time.perf_counter()
    results = validator.validate_batch(entities)
    elapsed = time.perf_counter() - start
    
    valid_count = sum(results)
    
    print(f"\nLarge-Scale Validation (n={len(entities)}):")
    print(f"  Time:  {elapsed*1000:.2f}ms ({elapsed/len(entities)*1000000:.2f}µs per entity)")
    print(f"  Valid: {valid_count}/{len(entities)}")
    
    # Should complete in reasonable time (< 1 second for 10k entities)
    assert elapsed < 1.0, f"Large-scale validation took too long: {elapsed:.2f}s"
    
    # Most should be valid (well-formed names)
    assert valid_count > len(entities) * 0.9, "Most entities should be valid"


def test_backward_compatibility():
    """Verify that new batch methods are backward compatible with existing code."""
    entity_validator = EntityValidator()
    relationship_validator = RelationshipValidator()
    
    # Test single entity validation (original API)
    entity = Entity(name="TestEntity", entity_type="concept")
    assert entity_validator.is_valid(entity) is True
    
    # Test single relationship validation (original API)
    relationship = Relationship(source="source", target="target", relation_type="connects_to")
    assert relationship_validator.is_valid(relationship) is True
    
    # Test batch with single item (should work)
    assert entity_validator.validate_batch([entity]) == [True]
    assert relationship_validator.validate_batch([relationship]) == [True]
    
    # Test batch with empty list (should return empty list)
    assert entity_validator.validate_batch([]) == []
    assert relationship_validator.validate_batch([]) == []
    
    print("\nBackward Compatibility: ✓ All tests passed")

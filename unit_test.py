from functools import wraps

class TestAssertions:
    """Custom assertion methods with detailed reporting"""
    
    @staticmethod
    def assert_eq(actual, expected, message=""):
        if actual != expected:
            error_msg = f"Expected {repr(expected)}, but got {repr(actual)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
    
    @staticmethod
    def assert_neq(actual, expected, message=""):
        if actual == expected:
            error_msg = f"Expected values to be different, but both were {repr(actual)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
    
    @staticmethod
    def assert_true(condition, message=""):
        if not condition:
            error_msg = f"Expected True, but got {repr(condition)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
    
    @staticmethod
    def assert_in(item, container, message=""):
        if item not in container:
            error_msg = f"Expected {repr(item)} to be in {repr(container)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
    
    @staticmethod
    def assert_gt(actual, expected, message=""):
        if not actual > expected:
            error_msg = f"Expected {repr(actual)} > {repr(expected)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
    
    @staticmethod
    def assert_lt(actual, expected, message=""):
        if not actual < expected:
            error_msg = f"Expected {repr(actual)} < {repr(expected)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)

    @staticmethod
    def assert_geq(actual, expected, message=""):
        if not actual >= expected:
            error_msg = f"Expected {repr(actual)} >= {repr(expected)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
    
    @staticmethod
    def assert_leq(actual, expected, message=""):
        if not actual <= expected:
            error_msg = f"Expected {repr(actual)} <= {repr(expected)}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)

def unit_test(func):
    """Simple decorator with better error reporting"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Running test: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            print(f"✓ {func.__name__} passed")
            return result
        except AssertionError as e:
            print(f"✗ {func.__name__} failed:")
            print(f"  {e}")
            raise
        except Exception as e:
            print(f"✗ {func.__name__} failed with exception: {e}")
            print(e.with_traceback())
            raise
    
    wrapper._is_unit_test = True
    return wrapper

def run_unit_tests(test_class_instance):
    test_methods = []
    
    for attr_name in dir(test_class_instance):
        attr = getattr(test_class_instance, attr_name)
        
        if callable(attr) and hasattr(attr, '_is_unit_test'):
            test_methods.append((attr_name, attr))
    
    print(f"Found {len(test_methods)} unit tests")
    print("-" * 40)
    
    passed = 0
    failed = 0
    
    for _, method in test_methods:
        try:
            method() 
            passed += 1
        except Exception:
            failed += 1
    
    print("-" * 40)
    print(f"Tests run: {len(test_methods)}, Passed: {passed}, Failed: {failed}")
    
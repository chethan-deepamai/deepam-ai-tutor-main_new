try:
    import main
    print('Import successful')
    print('app attribute exists:', hasattr(main, 'app'))
    if hasattr(main, 'app'):
        print('app type:', type(main.app))
    else:
        print('Available attributes:', [attr for attr in dir(main) if not attr.startswith('_')])
except Exception as e:
    print('Import failed:', e)
    import traceback
    traceback.print_exc()

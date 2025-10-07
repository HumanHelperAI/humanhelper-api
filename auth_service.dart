// auth_service.dart
import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

class AuthService {
  final String base = 'https://api.humanhelperai.in';

  String? _access;
  String? _refresh;
  bool _refreshing = false;
  final List<Completer<String?>> _waiters = [];

  Map<String, String> _headers({bool auth = true}) {
    final h = {'Content-Type': 'application/json'};
    if (auth && _access != null) h['Authorization'] = 'Bearer $_access';
    return h;
  }

  Future<Map<String, dynamic>> _post(String path, Map body, {bool auth = true}) async {
    final uri = Uri.parse('$base$path');
    final r = await http.post(uri, headers: _headers(auth: auth), body: jsonEncode(body));
    return _decodeOrThrow(r);
  }

  Map<String, dynamic> _decodeOrThrow(http.Response r) {
    final m = jsonDecode(r.body.isEmpty ? '{}' : r.body);
    if (r.statusCode >= 200 && r.statusCode < 300) return m;
    throw HttpException(r.statusCode, m);
  }

  // --- public API ---
  Future<Map<String, dynamic>> register({
    required String fullName,
    required String mobile,
    required String password,
    String? email,
    required String address,
  }) => _post('/auth/register', {
        'full_name': fullName,
        'mobile': mobile,
        'password': password,
        'email': email ?? '',
        'address': address,
      }, auth: false);

  Future<Map<String, dynamic>> verify({
    required String mobile,
    required String code,
  }) => _post('/auth/verify', {'mobile': mobile, 'code': code}, auth: false);

  Future<Map<String, dynamic>> resendCode({required String mobile}) =>
      _post('/auth/resend-code', {'mobile': mobile}, auth: false);

  Future<Map<String, dynamic>> login({
    required String mobile,
    required String password,
  }) async {
    final m = await _post('/auth/login', {'mobile': mobile, 'password': password}, auth: false);
    _access = m['access'];
    _refresh = m['refresh'];
    return m;
  }

  Future<Map<String, dynamic>> whoami() async {
    try {
      final uri = Uri.parse('$base/whoami');
      final r = await http.get(uri, headers: _headers());
      if (r.statusCode == 401) {
        // try refresh & retry once
        final ok = await _ensureFreshAccess();
        if (!ok) throw HttpException(401, {'error': 'invalid or expired token'});
        final r2 = await http.get(uri, headers: _headers());
        return _decodeOrThrow(r2);
      }
      return _decodeOrThrow(r);
    } on HttpException {
      rethrow;
    }
  }

  Future<void> logout() async {
    if (_refresh != null) {
      try { await _post('/auth/logout', {'refresh': _refresh}, auth: false); } catch (_) {}
    }
    _access = null;
    _refresh = null;
  }

  // --- refresh logic (single-flight) ---
  Future<bool> _ensureFreshAccess() async {
    if (_refresh == null) return false;

    if (_refreshing) {
      final c = Completer<String?>();
      _waiters.add(c);
      final v = await c.future;
      return v != null;
    }

    _refreshing = true;
    try {
      final m = await _post('/auth/refresh', {'refresh': _refresh}, auth: false);
      _access = m['access'] as String?;
      _refresh = (m['refresh'] ?? _refresh) as String?;
      for (final w in _waiters) w.complete(_access);
      _waiters.clear();
      return _access != null;
    } catch (e) {
      for (final w in _waiters) w.complete(null);
      _waiters.clear();
      _access = null;
      _refresh = null;
      return false;
    } finally {
      _refreshing = false;
    }
  }
}

class HttpException implements Exception {
  final int status;
  final Map<String, dynamic> body;
  HttpException(this.status, this.body);
  @override
  String toString() => 'HttpException($status, $body)';
}

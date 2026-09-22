// Сверяет JavaScript-движок с эталоном из Python.
//
// Движок генерируется, но генератор тоже можно сломать. Этот файл
// прогоняет обе реализации на одних и тех же фразах и требует
// совпадения уровня тревоги. Расхождение означает, что на телефоне
// правила ведут себя не так, как проверено тестами.
//
// Запуск: node tools/check_engine_parity.js <json со списком фраз>

const fs = require("fs");
const path = require("path");
const { evaluate, LEVELS } = require(path.join(__dirname, "..", "web", "engine.js"));

const expected = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let failed = 0;

for (const item of expected) {
  const got = evaluate(item.text).level;
  const ok = got === item.level;
  if (!ok) {
    failed++;
    console.log(`РАСХОЖДЕНИЕ: «${item.text.slice(0, 60)}»`);
    console.log(`  python: ${item.level}   javascript: ${got}`);
  }
}

console.log(failed === 0
  ? `совпало на всех ${expected.length} фразах`
  : `расхождений: ${failed} из ${expected.length}`);
process.exit(failed === 0 ? 0 : 1);

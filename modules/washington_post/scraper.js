const puppeteer = require('puppeteer')
const cheerio = require('cheerio')
const ObjectsToCsv = require('objects-to-csv');

const baseUrl =
	'https://www.washingtonpost.com/search/?query=ukraine+war&facets=%7B%22time%22%3A%22all%22%2C%22sort%22%3A%22relevancy%22%2C%22section%22%3A%5B%5D%2C%22author%22%3A%5B%5D%7D'


function delay(time) {
	return new Promise(function (resolve) {
		setTimeout(resolve, time)
	})
}

const main = async () => {
	const browser = await puppeteer.launch({ headless: false })
	const page = await browser.newPage()

	await page.goto(baseUrl, { waitUntil: 'networkidle2' })

	const articleClassName = '.single-result'
	await page.waitForSelector(articleClassName)
	let bodyHandle = await page.$('body')
	let html = await page.evaluate((body) => body.innerHTML, bodyHandle)
	let $ = cheerio.load(html)

	for (let i = 0; ; ++i) {
		console.log(
			$(
				` article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(1) > div`,
			).length,
		)
		const button = await page.$(
			'#main-content > div.jsx-2309075444.search-app-container.mr-auto.ml-auto.flex.flex-column.col-8-lg > section.jsx-2865089505.search-results-wrapper > button',
		)
		if (!button) break
		await button.evaluate((butt) => butt.click())
		await delay(1500)
		// await page.waitForResponse((res) => {
		// 	return res.status() === 200
		// })
		bodyHandle = await page.$('body')
		html = await page.evaluate((body) => body.innerHTML, bodyHandle)
		$ = cheerio.load(html)
		console.log(
			$(
				` article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(1) > div`,
			).length,
		)
	}

	const articles = []
	$(
		` article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(1) > div.font-body.font-light.font-xxs.antialiased.gray-dark.lh-1.mt-sm.mb-sm`,
	).each(function (i, elem) {
		articles.push({ id: i, claim: $(this).text(), truth_value: 'meter-true' })
	})

	$(
		` article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(1) > a`,
	).each(function (i, elem) {
		articles[i]['title'] = $(this).text()
	})

	$(
		` article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(2) > span`,
	).each(function (i, elem) {
		articles[i]['stated_on'] = $(this).text()
	})
	const csv = new ObjectsToCsv(articles);

	// Save to file:
	await csv.toDisk('./washington_post.csv');
	await browser.close()
}

main()
